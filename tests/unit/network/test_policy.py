"""The politeness policy: throttle, retry with backoff, timeout, cache, logging.

The policy is deliberately ignorant of transport. It takes a callable that
performs the request, so a vendor SDK goes through it unchanged — configuring
retry on the SDK instead would be the same bug in a new costume, two places to
remember politeness.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import pytest
import requests

from src.config import ConfigError, NetworkConfig, NetworkPolicy, Patience
from src.network.policy import RequestPolicy
from src.network.protocols import DO_NOT_RETRY, RETRY_WITH_BACKOFF, RetryAdvice
from src.network.throttle import InMemoryThrottle

HOST = "api.example.com"
START = datetime(2026, 8, 17, 2, 0, tzinfo=timezone.utc)


class FakeClock:
    def __init__(self) -> None:
        self.now = START
        self.slept: list[float] = []

    def get_now(self) -> datetime:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += timedelta(seconds=seconds)


class RecordingCache:
    """Returns what it was seeded with and records what it was handed.

    A spy, not a mirror: it makes no claim about how any real cache keys itself,
    which is the caller's business and differs per provider.
    """

    def __init__(self, stored: Any = None) -> None:
        self.stored = stored
        self.puts: list[Any] = []

    def get(self) -> Any:
        return self.stored

    def put(self, value: Any) -> None:
        self.puts.append(value)


class RecordingLogger:
    def __init__(self) -> None:
        self.messages: list[tuple[str, str]] = []

    def info(self, message: str, *, component: str = "", duration_ms: int = 0) -> None:
        self.messages.append(("info", message))

    def warning(self, message: str, *, component: str = "", duration_ms: int = 0) -> None:
        self.messages.append(("warning", message))

    def error(self, message: str, *, component: str = "", duration_ms: int = 0) -> None:
        self.messages.append(("error", message))


POLICY = "test_policy"


def _limits(
    *,
    min_interval_seconds: float = 1.0,
    timeout_seconds: float = 30.0,
    max_attempts: int = 3,
    backoff_base_seconds: float = 1.0,
    backoff_max_seconds: float = 60.0,
    cache_ttl: timedelta | None = timedelta(hours=1),
) -> NetworkPolicy:
    """A complete policy, for tests about one field at a time.

    `NetworkPolicy` itself has no defaults — a default is a policy nobody
    decided, applied to a host nobody considered. These live in the test because
    a test about backoff should not have to restate a timeout to say what it means.
    """
    return NetworkPolicy(
        min_interval_seconds=min_interval_seconds,
        timeout_seconds=timeout_seconds,
        max_attempts=max_attempts,
        backoff_base_seconds=backoff_base_seconds,
        backoff_max_seconds=backoff_max_seconds,
        cache_ttl=cache_ttl,
    )


def _patience(
    *,
    timeout_seconds: float = 300.0,
    max_attempts: int = 2,
    backoff_base_seconds: float = 5.0,
    backoff_max_seconds: float = 30.0,
) -> Patience:
    """A complete patience, for tests about one field at a time.

    It states no spacing and no lifetime: those describe the host, and a
    patience that could set them would be a per-request licence to out-pace
    another caller of the same server.
    """
    return Patience(
        timeout_seconds=timeout_seconds,
        max_attempts=max_attempts,
        backoff_base_seconds=backoff_base_seconds,
        backoff_max_seconds=backoff_max_seconds,
    )


def _network(**overrides: Any) -> NetworkConfig:
    """A config declaring one policy, with `HOST` assigned to it."""
    return NetworkConfig(
        policies={POLICY: _limits(**overrides)}, hosts={HOST: POLICY}
    )


def _never_transient(error: BaseException) -> RetryAdvice:
    return DO_NOT_RETRY


def _always_transient(error: BaseException) -> RetryAdvice:
    return RETRY_WITH_BACKOFF


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def logger() -> RecordingLogger:
    return RecordingLogger()


def _policy(clock: FakeClock, logger: RecordingLogger, network: NetworkConfig | None = None):
    return RequestPolicy(
        network=network or _network(),
        throttle=InMemoryThrottle(get_now=clock.get_now, sleep=clock.sleep),
        sleep=clock.sleep,
        random=lambda: 1.0,
        logger=logger,
    )


class TestTheCacheComesFirst:
    def test_a_hit_is_served_without_performing(self, clock, logger):
        performed: list[float] = []

        result = _policy(clock, logger).call(
            host=HOST,
            perform=lambda timeout: performed.append(timeout) or "fresh",
            is_transient=_never_transient,
            cache=RecordingCache(stored="remembered"),
            label="example",
        )

        assert result == "remembered"
        assert performed == []

    def test_a_hit_does_not_spend_the_hosts_interval(self, clock, logger):
        """A cache exists to remove the call, so it must not also pay for one.

        Throttling a hit would make a warm cache as slow as a cold one and
        silently serialise a batch that is doing no network work at all.
        """
        policy = _policy(clock, logger)
        cache = RecordingCache(stored="remembered")

        policy.call(host=HOST, perform=lambda t: "fresh", is_transient=_never_transient,
                    cache=cache, label="example")
        policy.call(host=HOST, perform=lambda t: "fresh", is_transient=_never_transient,
                    cache=cache, label="example")

        assert clock.slept == []

    def test_a_miss_stores_what_it_fetched(self, clock, logger):
        cache = RecordingCache(stored=None)

        _policy(clock, logger).call(
            host=HOST, perform=lambda t: "fresh", is_transient=_never_transient,
            cache=cache, label="example",
        )

        assert cache.puts == ["fresh"]


class TestTheTimeoutReachesTheCall:
    def test_perform_is_handed_the_hosts_timeout(self, clock, logger):
        """A timeout cannot be imposed on an opaque callable from outside, so
        the policy hands its number in and the transport applies it."""
        network = _network(timeout_seconds=45.0)
        seen: list[float] = []

        _policy(clock, logger, network).call(
            host=HOST, perform=lambda timeout: seen.append(timeout),
            is_transient=_never_transient, cache=RecordingCache(), label="example",
        )

        assert seen == [45.0]

    def test_each_host_gets_its_own_timeout(self, clock, logger):
        """A slow venue site and an API are not owed the same patience."""
        network = NetworkConfig(
            policies={
                "api": _limits(timeout_seconds=5.0),
                "web": _limits(timeout_seconds=60.0),
            },
            hosts={HOST: "api", "slow-venue.example": "web"},
        )
        policy = _policy(clock, logger, network)
        seen: list[float] = []

        for host in (HOST, "slow-venue.example"):
            policy.call(
                host=host, perform=lambda timeout: seen.append(timeout),
                is_transient=_never_transient, cache=RecordingCache(), label="example",
            )

        assert seen == [5.0, 60.0]

    def test_an_undeclared_host_is_refused_at_the_call(self, clock, logger):
        """The config error surfaces where the call is made, naming the host.

        A guess here would be the failure the whole section exists to prevent:
        a host nobody set a policy for, quietly borrowing somebody else's.
        """
        with pytest.raises(ConfigError, match="undeclared.example"):
            _policy(clock, logger).call(
                host="undeclared.example", perform=lambda timeout: "fresh",
                is_transient=_never_transient, cache=RecordingCache(), label="example",
            )


class TestACategoryNamedAtTheCallSite:
    """Hosts that arrive from fetched data cannot be assigned in advance.

    An image URL points at whatever CDN a venue uses, so the caller names the
    policy it was built to use. The throttle still spaces per host — it just
    reads its numbers from a policy nobody had to enumerate hosts for.
    """

    def test_a_named_policy_is_used_instead_of_a_host_assignment(self, clock, logger):
        network = NetworkConfig(
            policies={"data_derived": _limits(timeout_seconds=7.0)}, hosts={}
        )
        seen: list[float] = []

        _policy(clock, logger, network).call(
            host="cdn.unknown.example", policy="data_derived",
            perform=lambda timeout: seen.append(timeout),
            is_transient=_never_transient, cache=RecordingCache(), label="image",
        )

        assert seen == [7.0]

    def test_spacing_is_still_per_host(self, clock, logger):
        """Two images from one CDN wait; two from different CDNs do not.

        Sharing a policy must not collapse into sharing a queue, or every image
        in a batch would serialise behind whichever CDN was slowest.
        """
        network = NetworkConfig(
            policies={"data_derived": _limits(min_interval_seconds=2.0)}, hosts={}
        )
        policy = _policy(clock, logger, network)

        for host in ("cdn-a.example", "cdn-b.example", "cdn-a.example"):
            policy.call(
                host=host, policy="data_derived", perform=lambda timeout: "bytes",
                is_transient=_never_transient, cache=RecordingCache(), label="image",
            )

        assert clock.slept == [pytest.approx(2.0)]

    def test_an_undeclared_policy_name_is_refused(self, clock, logger):
        with pytest.raises(ConfigError, match="images"):
            _policy(clock, logger).call(
                host="cdn.unknown.example", policy="images",
                perform=lambda timeout: "bytes",
                is_transient=_never_transient, cache=RecordingCache(), label="image",
            )


class TestPatienceIsNamedByTheRequest:
    """Spacing is the host's; how long to wait for an answer is the request's.

    A generation takes minutes whoever runs it; an embedding takes milliseconds
    whoever runs it. One host answering both — Ollama on `/api/chat` and
    `/api/embed`, or a hosted provider serving generation and embeddings from
    one address — is the case a single per-host policy cannot express, and the
    reason this is not a second policy named `local_model_extraction`.

    A caller naming no patience is unaffected, which is what let this arrive
    without touching the eight callers that already existed.
    """

    def _network(self, **patience_overrides) -> NetworkConfig:
        return NetworkConfig(
            policies={POLICY: _limits(min_interval_seconds=0.0, timeout_seconds=30.0)},
            hosts={HOST: POLICY},
            patience={"generation": _patience(**patience_overrides)},
        )

    def test_the_named_patience_supplies_the_timeout(self, clock, logger):
        seen: list[float] = []

        _policy(clock, logger, self._network(timeout_seconds=900.0)).call(
            host=HOST, patience="generation",
            perform=lambda timeout: seen.append(timeout),
            is_transient=_never_transient, cache=RecordingCache(), label="extraction",
        )

        assert seen == [900.0]

    def test_naming_none_leaves_the_host_in_charge(self, clock, logger):
        """The existing callers must not change behaviour by standing still."""
        seen: list[float] = []

        _policy(clock, logger, self._network(timeout_seconds=900.0)).call(
            host=HOST, perform=lambda timeout: seen.append(timeout),
            is_transient=_never_transient, cache=RecordingCache(), label="embedding",
        )

        assert seen == [30.0]

    def test_the_named_patience_supplies_the_attempts(self, clock, logger):
        attempts: list[int] = []

        network = NetworkConfig(
            policies={POLICY: _limits(min_interval_seconds=0.0, max_attempts=5)},
            hosts={HOST: POLICY},
            patience={"generation": _patience(max_attempts=2)},
        )

        def perform(timeout: float) -> str:
            attempts.append(1)
            raise RuntimeError("503 unavailable")

        with pytest.raises(RuntimeError):
            _policy(clock, logger, network).call(
                host=HOST, patience="generation", perform=perform,
                is_transient=_always_transient, cache=RecordingCache(), label="extraction",
            )

        assert len(attempts) == 2

    def test_the_named_patience_supplies_the_backoff(self, clock, logger):
        network = NetworkConfig(
            policies={POLICY: _limits(min_interval_seconds=0.0, backoff_base_seconds=1.0)},
            hosts={HOST: POLICY},
            patience={"generation": _patience(max_attempts=2, backoff_base_seconds=5.0)},
        )

        with pytest.raises(RuntimeError):
            _policy(clock, logger, network).call(
                host=HOST, patience="generation", perform=_raising,
                is_transient=_always_transient, cache=RecordingCache(), label="extraction",
            )

        assert clock.slept == [pytest.approx(5.0)]

    def test_the_hosts_spacing_survives_being_named(self, clock, logger):
        """Patience says nothing about spacing, so it cannot make us impolite.

        The case that matters is a hosted provider: name `generation` against
        it and it still waits the interval its host was assigned.
        """
        network = NetworkConfig(
            policies={POLICY: _limits(min_interval_seconds=2.0)},
            hosts={HOST: POLICY},
            patience={"generation": _patience()},
        )
        policy = _policy(clock, logger, network)

        for _ in range(2):
            policy.call(
                host=HOST, patience="generation", perform=lambda timeout: "answer",
                is_transient=_never_transient, cache=RecordingCache(), label="extraction",
            )

        assert clock.slept == [pytest.approx(2.0)]

    def test_an_undeclared_patience_is_refused_before_anything_is_asked(
        self, clock, logger
    ):
        performed: list[float] = []

        with pytest.raises(ConfigError, match="transcription"):
            _policy(clock, logger, self._network()).call(
                host=HOST, patience="transcription",
                perform=lambda timeout: performed.append(timeout),
                is_transient=_never_transient, cache=RecordingCache(), label="whisper",
            )

        assert performed == []


class TestRetry:
    def test_a_transient_failure_is_tried_again(self, clock, logger):
        attempts: list[int] = []

        def perform(timeout: float) -> str:
            attempts.append(1)
            if len(attempts) < 3:
                raise RuntimeError("503")
            return "fresh"

        result = _policy(clock, logger).call(
            host=HOST, perform=perform, is_transient=_always_transient,
            cache=RecordingCache(), label="example",
        )

        assert result == "fresh"
        assert len(attempts) == 3

    def test_a_permanent_failure_is_not_tried_again(self, clock, logger):
        """A bad request repeated politely is still a bad request."""
        attempts: list[int] = []

        def perform(timeout: float) -> str:
            attempts.append(1)
            raise RuntimeError("404")

        with pytest.raises(RuntimeError, match="404"):
            _policy(clock, logger).call(
                host=HOST, perform=perform, is_transient=_never_transient,
                cache=RecordingCache(), label="example",
            )

        assert len(attempts) == 1

    def test_exhausting_the_attempts_raises_the_last_error(self, clock, logger):
        network = _network(max_attempts=3)
        attempts: list[int] = []

        def perform(timeout: float) -> str:
            attempts.append(1)
            raise RuntimeError(f"attempt {len(attempts)}")

        with pytest.raises(RuntimeError, match="attempt 3"):
            _policy(clock, logger, network).call(
                host=HOST, perform=perform, is_transient=_always_transient,
                cache=RecordingCache(), label="example",
            )

        assert len(attempts) == 3

    def test_a_failed_call_stores_nothing(self, clock, logger):
        """Caching a failure would serve it back for the whole TTL."""
        cache = RecordingCache()

        with pytest.raises(RuntimeError):
            _policy(clock, logger).call(
                host=HOST, perform=_raising, is_transient=_never_transient,
                cache=cache, label="example",
            )

        assert cache.puts == []

    def test_every_attempt_is_throttled_not_just_the_first(self, clock, logger):
        """A retry storm is the least polite moment to stop being polite.

        Acquiring once outside the loop would let a failing host be hammered
        `max_attempts` times back to back with only the backoff between.
        """
        network = _network(
                max_attempts=2, min_interval_seconds=4.0,
                backoff_base_seconds=1.0, backoff_max_seconds=60.0,
            )

        with pytest.raises(RuntimeError):
            _policy(clock, logger, network).call(
                host=HOST, perform=_raising, is_transient=_always_transient,
                cache=RecordingCache(), label="example",
            )

        # 1.0s of backoff, then the throttle waiting out the rest of its 4s.
        assert clock.slept == [pytest.approx(1.0), pytest.approx(3.0)]


class TestBackoff:
    def test_it_grows_exponentially(self, clock, logger):
        network = _network(
                max_attempts=4, min_interval_seconds=0.0,
                backoff_base_seconds=1.0, backoff_max_seconds=60.0,
            )

        with pytest.raises(RuntimeError):
            _policy(clock, logger, network).call(
                host=HOST, perform=_raising, is_transient=_always_transient,
                cache=RecordingCache(), label="example",
            )

        assert clock.slept == [pytest.approx(1.0), pytest.approx(2.0), pytest.approx(4.0)]

    def test_it_is_capped(self, clock, logger):
        """Uncapped, a dead host stalls the batch for hours instead of failing."""
        network = _network(
                max_attempts=5, min_interval_seconds=0.0,
                backoff_base_seconds=10.0, backoff_max_seconds=25.0,
            )

        with pytest.raises(RuntimeError):
            _policy(clock, logger, network).call(
                host=HOST, perform=_raising, is_transient=_always_transient,
                cache=RecordingCache(), label="example",
            )

        assert clock.slept == [
            pytest.approx(10.0), pytest.approx(20.0),
            pytest.approx(25.0), pytest.approx(25.0),
        ]

    def test_jitter_never_drops_below_half_the_backoff(self, clock, logger):
        """Full jitter can round to no wait at all, which is impolite at exactly
        the moment the server asked for room."""
        network = _network(
                max_attempts=2, min_interval_seconds=0.0,
                backoff_base_seconds=8.0, backoff_max_seconds=60.0,
            )
        policy = RequestPolicy(
            network=network,
            throttle=InMemoryThrottle(get_now=clock.get_now, sleep=clock.sleep),
            sleep=clock.sleep,
            random=lambda: 0.0,
            logger=logger,
        )

        with pytest.raises(RuntimeError):
            policy.call(host=HOST, perform=_raising, is_transient=_always_transient,
                        cache=RecordingCache(), label="example")

        assert clock.slept == [pytest.approx(4.0)]

    def test_retry_after_overrides_the_computed_backoff(self, clock, logger):
        """The server named a number; honour it rather than guessing past it."""
        network = _network(
                max_attempts=2, min_interval_seconds=0.0,
                backoff_base_seconds=1.0, backoff_max_seconds=60.0,
            )

        def advises_retry_after(error: BaseException) -> RetryAdvice:
            return RetryAdvice(retry=True, retry_after_seconds=17.0)

        with pytest.raises(RuntimeError):
            _policy(clock, logger, network).call(
                host=HOST, perform=_raising, is_transient=advises_retry_after,
                cache=RecordingCache(), label="example",
            )

        assert clock.slept == [pytest.approx(17.0)]

    def test_a_retry_after_shorter_than_our_own_floor_is_still_waited_out(self, clock, logger):
        """A server asking for less than we promised does not license us to be
        quicker than we promised.

        Nominatim publishes one request a second and `min_interval_seconds` is
        1.5 — deliberately more restraint than it demands. Honouring a
        `Retry-After: 0` literally would spend that choice the moment a host
        happened to be having a bad minute, which is the worst moment to have it.
        The gap is `max(what it asked for, what we owe)`, because the throttle
        gates every attempt rather than only the first.
        """
        network = _network(
                max_attempts=2, min_interval_seconds=1.5,
                backoff_base_seconds=1.0, backoff_max_seconds=60.0,
            )

        with pytest.raises(RuntimeError):
            _policy(clock, logger, network).call(
                host=HOST, perform=_raising,
                is_transient=lambda e: RetryAdvice(retry=True, retry_after_seconds=0.0),
                cache=RecordingCache(), label="example",
            )

        # Asserted as a total rather than a list: what the host experiences is
        # the gap between the two requests, and how that is split between the
        # backoff sleep and the throttle's is an implementation detail.
        assert sum(clock.slept) == pytest.approx(1.5)

    def test_retry_after_is_not_jittered(self, clock, logger):
        """Jittering a number the server chose is second-guessing it."""
        network = _network(
                max_attempts=2, min_interval_seconds=0.0,
                backoff_base_seconds=1.0, backoff_max_seconds=60.0,
            )
        policy = RequestPolicy(
            network=network,
            throttle=InMemoryThrottle(get_now=clock.get_now, sleep=clock.sleep),
            sleep=clock.sleep,
            random=lambda: 0.0,
            logger=logger,
        )

        with pytest.raises(RuntimeError):
            policy.call(
                host=HOST, perform=_raising,
                is_transient=lambda e: RetryAdvice(retry=True, retry_after_seconds=17.0),
                cache=RecordingCache(), label="example",
            )

        assert clock.slept == [pytest.approx(17.0)]


class TestLogging:
    def test_a_retry_says_who_was_asked_and_how_long_we_waited(self, clock, logger):
        """One place that can answer "what did we ask whom, and how often"."""
        network = _network(max_attempts=2, min_interval_seconds=0.0)

        with pytest.raises(RuntimeError):
            _policy(clock, logger, network).call(
                host=HOST, perform=_raising, is_transient=_always_transient,
                cache=RecordingCache(), label="example",
            )

        retries = [m for level, m in logger.messages if level == "warning"]
        assert len(retries) == 1
        assert HOST in retries[0]


class TestWhatTheProviderSaid:
    """A refusal should say why, not only that.

    Open-Meteo answered 400 on every batch run for two days and the log recorded
    `400 Client Error: Bad Request for url: ...` — the question, never the
    answer. The reason was in the response body the whole time, and reading it
    meant re-issuing the request by hand:

        {"error":true,"reason":"Parameter 'start_date' is out of allowed range
                                from 2026-05-18 to 2026-09-03"}

    Logged, that one line names both halves — the date we asked for is already
    in the URL — and the bound is diagnosed on the night it broke.

    Deliberately **not** parsed. A provider's error format is theirs to change,
    and a bound is a judgement made when the call site is written, never
    something the code revises at runtime from an error message.
    """

    def test_the_response_body_is_logged_with_the_failure(self, clock, logger):
        reason = '{"error":true,"reason":"start_date is out of allowed range"}'

        self._give_up(clock, logger, _raising_with_response(reason))

        assert reason in self._failure(logger)

    def test_a_real_requests_error_carries_its_body_through(self, clock, logger):
        """The shape this exists for, from the library that actually raises it.

        The policy is transport-agnostic by design and reads the response
        defensively, so the duck-typed case above is its real contract. This
        pins the one implementation that matters against the genuine object,
        because a defensive read that quietly finds nothing looks identical to
        a provider that said nothing.
        """
        response = requests.Response()
        response.status_code = 400
        response._content = b'{"reason":"out of allowed range"}'
        error = requests.HTTPError("400 Client Error: Bad Request", response=response)

        self._give_up(clock, logger, _raising_error(error))

        assert "out of allowed range" in self._failure(logger)

    def test_a_failure_carrying_no_response_logs_as_it_always_did(self, clock, logger):
        """A timeout and a DNS error have no body, and must not acquire one.

        The regression case: the policy wraps arbitrary callables, so `response`
        is an attribute it may not find rather than one it can assume.
        """
        self._give_up(clock, logger, _raising)

        message = self._failure(logger)
        assert "503 unavailable" in message
        assert "said" not in message

    def test_an_empty_body_adds_nothing(self, clock, logger):
        """Nothing to report reads as nothing, not as an empty quotation."""
        self._give_up(clock, logger, _raising_with_response("   "))

        assert "said" not in self._failure(logger)

    def test_a_long_body_is_truncated(self, clock, logger):
        """An HTML error page must not flood a batch log."""
        self._give_up(clock, logger, _raising_with_response("x" * 5_000))

        message = self._failure(logger)
        assert len(message) < 1_000
        assert message.endswith("…")

    def test_a_body_is_flattened_onto_one_line(self, clock, logger):
        """The log is one JSON object a line, and stays readable as one."""
        self._give_up(clock, logger, _raising_with_response("first\n\n  second"))

        assert "first second" in self._failure(logger)

    def test_a_retry_does_not_repeat_the_body(self, clock, logger):
        """Three attempts should not mean three copies of the same error page.

        The give-up line is the one a person reads. A retry line says only that
        we are trying again, which needs no explanation from the provider.
        """
        network = _network(max_attempts=2, min_interval_seconds=0.0)

        with pytest.raises(_RefusedWithBody):
            _policy(clock, logger, network).call(
                host=HOST, perform=_raising_with_response("the reason"),
                is_transient=_always_transient, cache=RecordingCache(),
                label="example",
            )

        retries = [m for level, m in logger.messages if level == "warning"]
        assert retries and all("the reason" not in line for line in retries)

    @staticmethod
    def _give_up(clock, logger, perform) -> None:
        with pytest.raises(Exception):
            _policy(clock, logger, _network(max_attempts=1)).call(
                host=HOST, perform=perform, is_transient=_never_transient,
                cache=RecordingCache(), label="example",
            )

    @staticmethod
    def _failure(logger) -> str:
        failures = [m for level, m in logger.messages if level == "error"]
        assert len(failures) == 1
        return failures[0]


def _raising(timeout: float) -> str:
    raise RuntimeError("503 unavailable")


class _RefusedWithBody(Exception):
    """An error carrying a response, as every HTTP library's does."""

    def __init__(self, body: str) -> None:
        super().__init__("400 Client Error: Bad Request")
        self.response = _ResponseWithText(body)


class _ResponseWithText:
    def __init__(self, text: str) -> None:
        self.text = text


def _raising_with_response(body: str):
    def perform(timeout: float) -> str:
        raise _RefusedWithBody(body)

    return perform


def _raising_error(error: BaseException):
    def perform(timeout: float) -> str:
        raise error

    return perform
