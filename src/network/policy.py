"""The one place politeness lives, for every call to something we do not run.

`CLAUDE.md` working agreement 2 says every outbound call owes three things: a
cache with a considered TTL, a bound on what is even asked, and a throttle. This
owns the first and third. **The second stays with the caller** — only it knows
that its provider cannot answer for day ninety, and the adapter making the ask
cheap is the wrong fix for a request that can never be answered.

It wraps a *call*, not a URL, so a vendor SDK goes through it unchanged.
"""

from __future__ import annotations

from typing import Callable, TypeVar

from src.config import NetworkConfig, NetworkPolicy
from src.network.protocols import CacheStrategy, TransientCheck
from src.network.throttle import Throttle
from src.utils.logging import StructuredLogger

T = TypeVar("T")


class RequestPolicy:
    """Throttle, retry with backoff, bound the wait, and say what happened."""

    def __init__(
        self,
        *,
        network: NetworkConfig,
        throttle: Throttle,
        sleep: Callable[[float], None],
        random: Callable[[], float],
        logger: StructuredLogger | None = None,
    ) -> None:
        """
        Args:
            network: The declared policies and their host assignments.
            throttle: Shared across every provider, because two callers of one
                host are one conversation from the server's side.
            sleep: Injected delay, so tests never wait in real time.
            random: Injected source of jitter, in `[0, 1)`. Required rather than
                defaulted to `random.random`: a default only production reaches
                is untested by construction.
            logger: Structured logger. Optional.
        """
        self._network = network
        self._throttle = throttle
        self._sleep = sleep
        self._random = random
        self._logger = logger

    def call(
        self,
        *,
        host: str,
        perform: Callable[[float], T],
        is_transient: TransientCheck,
        cache: CacheStrategy[T],
        label: str,
        policy: str | None = None,
    ) -> T:
        """Perform a request politely, or serve what the caller already had.

        Args:
            host: What is being asked. Throttling is per host, so two providers
                pointed at one service space against each other — and two images
                from one CDN wait for each other while two CDNs do not.
            perform: Does the request, given the timeout in seconds. The timeout
                is handed in rather than imposed, because it cannot be applied
                to an opaque callable from outside.
            is_transient: Reads this transport's failures.
            cache: Already bound to this call's key, with its own TTL.
            label: Source name, for log messages.
            policy: Names the policy directly, for a caller whose hosts are not
                knowable in advance — an image URL points at whatever CDN a venue
                uses, so those hosts cannot be assigned in config. Omitted, the
                host's own assignment is used, and an unassigned host is refused.

        Returns:
            The cached value when there is a fresh one, otherwise the result of
            the call.

        Raises:
            ConfigError: If the host has no assigned policy, or `policy` names
                one that is not declared.
            BaseException: Whatever `perform` raised on its final attempt.
        """
        cached = cache.get()
        if cached is not None:
            return cached

        limits = (
            self._network.for_category(policy)
            if policy is not None
            else self._network.for_host(host)
        )
        last_error: BaseException | None = None

        for attempt in range(1, limits.max_attempts + 1):
            self._throttle.acquire(
                host, min_interval_seconds=limits.min_interval_seconds
            )

            try:
                result = perform(limits.timeout_seconds)
            except BaseException as error:  # noqa: BLE001 - re-raised below
                last_error = error
                advice = is_transient(error)

                if not advice.retry or attempt == limits.max_attempts:
                    self._log_giving_up(host, label, attempt, error)
                    raise

                wait = self._wait_before(attempt, advice.retry_after_seconds, limits)
                self._log_retry(host, label, attempt, wait, error)
                self._sleep(wait)
                continue

            cache.put(result)
            return result

        raise AssertionError(  # pragma: no cover - the loop always returns or raises
            f"unreachable: {label} exhausted its attempts without an outcome"
        ) from last_error

    def _wait_before(
        self, attempt: int, retry_after_seconds: float | None, limits: NetworkPolicy
    ) -> float:
        """How long to wait before the next attempt.

        A `Retry-After` is honoured exactly and never jittered — the server named
        a number, and jittering it is second-guessing it. Otherwise the wait
        doubles per attempt, capped, then takes half-jitter: full jitter can
        round to no wait at all, which is impolite at exactly the moment the
        server asked for room.
        """
        if retry_after_seconds is not None:
            return retry_after_seconds

        backoff = min(
            limits.backoff_base_seconds * (2.0 ** (attempt - 1)),
            limits.backoff_max_seconds,
        )
        return backoff * (0.5 + 0.5 * self._random())

    def _log_retry(
        self, host: str, label: str, attempt: int, wait: float, error: BaseException
    ) -> None:
        if self._logger is None:
            return
        self._logger.warning(
            f"{label}: {host} attempt {attempt} failed ({error}); "
            f"retrying in {wait:.1f}s",
            component="network",
            duration_ms=0,
        )

    def _log_giving_up(
        self, host: str, label: str, attempt: int, error: BaseException
    ) -> None:
        if self._logger is None:
            return
        self._logger.error(
            f"{label}: {host} failed after {attempt} attempt(s) ({error})"
            f"{_what_the_provider_said(error)}",
            component="network",
            duration_ms=0,
        )


#: How much of a refused response to keep. Long enough for a JSON error object
#: — the ones worth reading state a reason and a permitted range in well under
#: this — and short enough that an HTML error page cannot flood a batch log.
MAX_LOGGED_BODY = 500


def _what_the_provider_said(error: BaseException) -> str:
    """The response body behind a failure, ready to append to a log line.

    Worth having because an exception says what we asked and not what came
    back. Open-Meteo refused one date on every run for two nights and logged
    `400 Client Error: Bad Request for url: ...`; the body said *"Parameter
    'start_date' is out of allowed range from 2026-05-18 to 2026-09-03"*, which
    is the entire diagnosis, and reading it meant re-issuing the request by
    hand.

    Reported verbatim and never parsed. A provider's error format belongs to
    the provider, and a bound is a judgement made where the call site is
    written — not something the code revises at runtime from an error message.

    Read defensively: this policy wraps arbitrary callables including vendor
    SDKs, so `response` is an attribute an error may happen to carry, never one
    it can be assumed to have.

    Returns:
        A parenthetical to append, or an empty string when there is no body —
        which reads as nothing at all rather than as an empty quotation.
    """
    response = getattr(error, "response", None)
    text = getattr(response, "text", None)
    if not isinstance(text, str) or not text.strip():
        return ""
    # Collapsed onto one line: the log is one JSON object per line, and an HTML
    # error page's newlines survive escaping to make it unreadable as one.
    flattened = " ".join(text.split())
    if len(flattened) > MAX_LOGGED_BODY:
        flattened = flattened[:MAX_LOGGED_BODY] + "…"
    return f" — the provider said: {flattened}"
