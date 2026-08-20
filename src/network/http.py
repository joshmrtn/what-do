"""The one part of politeness that really is about HTTP.

`RequestPolicy` deliberately inspects no URL and knows no transport, so that a
vendor SDK goes through it unchanged. Three things do not survive that
abstraction, and they live here:

* **which failures are worth another attempt** — a status code, whether it
  arrived as a `requests` exception or on a vendor SDK's error;
* **what the server said with `Retry-After`** — it named a number, so we use its
  number rather than our own schedule;
* **the conditional request** — replaying `ETag` and `Last-Modified` so a server
  can answer `304` and send no body at all, which is the politest outcome
  available short of not asking.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from typing import Any, Callable, Mapping
from urllib.parse import urlencode, urlsplit

import requests

from src.network.policy import RequestPolicy
from src.network.protocols import (
    DO_NOT_RETRY,
    RETRY_WITH_BACKOFF,
    RetryAdvice,
    TransientCheck,
)
from src.config import NetworkConfig
from src.storage.http_cache import CachedResponse
from src.storage.protocols import HttpCache
from src.utils.logging import StructuredLogger
from src.utils.secret import contains_secret

#: Identifies the project rather than impersonating a browser.
USER_AGENT = "what-do/1.0 (local event aggregator; nightly batch)"

#: What may go in a query string. Deliberately **not** `Mapping[str, Any]`: a
#: `Secret` is assignable to `Any`, and `urlencode` would then call `str()` on
#: it and send the redaction placeholder to the provider as the credential —
#: type-clean, test-clean, and wrong on the wire. Narrowed, `mypy --strict`
#: names the call site instead, which is the whole mechanism of the `Secret`
#: type. A caller that means to send one says `expose_secret()`.
QueryParams = Mapping[str, str | int | float]

#: Statuses that mean "not now" rather than "no". 429 is the server asking for
#: room; 5xx is it failing in a way that may not repeat. Everything else in the
#: 4xx range describes the request itself and will fail identically next time.
_RETRYABLE_STATUSES = frozenset({429})


def requests_transient_check(*, get_now: Callable[[], datetime]) -> TransientCheck:
    """A transient-failure predicate for `requests`.

    Args:
        get_now: Injected clock, needed because `Retry-After` may be an HTTP
            date rather than a count of seconds. Required rather than defaulted:
            a default only production reaches is untested by construction.

    Returns:
        A check reading `requests`' exceptions and any `Retry-After` on them.
    """

    def check(error: BaseException) -> RetryAdvice:
        if isinstance(error, requests.HTTPError):
            return _advice_for_http_error(error, get_now)
        if isinstance(error, (requests.Timeout, requests.ConnectionError)):
            return RETRY_WITH_BACKOFF
        # Anything else is not this transport's failure — most likely ours, and
        # a bug fails identically however politely it is repeated.
        return DO_NOT_RETRY

    return check


def api_status_transient_check(*, get_now: Callable[[], datetime]) -> TransientCheck:
    """A transient-failure predicate for an SDK error carrying an HTTP status.

    A vendor SDK does not raise `requests`' exceptions, but the service behind it
    still answers with a status, and the SDK keeps it: `google.genai`'s `APIError`
    carries `.code`, and the response it came from when there was one. A status
    means the same thing whoever transported it, so the judgement is the same and
    lives in one place.

    Read off the attribute rather than by type, so this module stays free of any
    vendor SDK — nothing in `src/network/` should have to be edited to add a
    provider.

    **It reads a status and nothing else.** An SDK whose transport failures carry
    no status — a timeout, a refused connection — needs those recognised too, and
    that vocabulary belongs to the provider: compose this with the provider's own
    check, as `gemini_transient_check` does.

    Args:
        get_now: Injected clock, for a `Retry-After` given as an HTTP date.

    Returns:
        A check reading `.code` and any `Retry-After` on the kept response.
    """

    def check(error: BaseException) -> RetryAdvice:
        status = getattr(error, "code", None)
        if not isinstance(status, int):
            return DO_NOT_RETRY
        response = getattr(error, "response", None)
        headers = getattr(response, "headers", None)
        return _advice_for_status(status, headers, get_now)

    return check


def _advice_for_http_error(
    error: requests.HTTPError, get_now: Callable[[], datetime]
) -> RetryAdvice:
    """Advice for a `requests` failure that reached a server.

    An error carrying no response has no status to judge, so it is not retried:
    without one there is nothing to call transient and guessing would retry the
    unretryable.
    """
    response = error.response
    if response is None:
        return DO_NOT_RETRY
    return _advice_for_status(response.status_code, response.headers, get_now)


def _advice_for_status(
    status: int,
    headers: Mapping[str, str] | None,
    get_now: Callable[[], datetime],
) -> RetryAdvice:
    """Advice for a status code, and the server's own opinion on when."""
    if status not in _RETRYABLE_STATUSES and status < 500:
        return DO_NOT_RETRY

    return RetryAdvice(
        retry=True,
        retry_after_seconds=_retry_after_seconds(
            headers.get("Retry-After") if headers is not None else None, get_now
        ),
    )


def _retry_after_seconds(
    header: str | None, get_now: Callable[[], datetime]
) -> float | None:
    """Read `Retry-After` in either legal form, or None for no opinion.

    Delta-seconds and an HTTP date are both valid (RFC 9110). Ignoring the date
    form would mean backing off two seconds against a server that asked for an
    hour, which is the impoliteness this module exists to prevent. Anything
    unparseable is treated as no opinion rather than as a reason to stop
    retrying — the status already said the attempt was worth repeating.

    Never negative: a stale date means "now", and a negative wait is not a wait.
    """
    if header is None:
        return None

    raw = header.strip()
    try:
        return max(0.0, float(raw))
    except ValueError:
        pass

    try:
        when = parsedate_to_datetime(raw)
    except (TypeError, ValueError):
        return None

    if when.tzinfo is None:
        return None

    return max(0.0, (when - get_now()).total_seconds())


@dataclass(frozen=True)
class HttpDocument:
    """A fetched body with whatever validators the server offered.

    The validators travel *with* the body rather than being written separately,
    because the policy caches whatever `perform` returned. Splitting them would
    leave the cache holding a body whose ETag was stored by a different code
    path — and then a 304 would revalidate against the wrong version.
    """

    body: str
    etag: str | None
    last_modified: str | None


class UrlCache:
    """A `CacheStrategy[HttpDocument]` bound to one URL.

    The freshness bound is handed in at construction rather than checked by the
    caller afterwards, so a stale read has no API. That is the same shape
    `DayCache.get(..., fresh_since=...)` uses, and for the same reason: a
    rule remembered at every call site is a rule that will be forgotten.
    """

    def __init__(
        self,
        *,
        url: str,
        http_cache: HttpCache,
        get_now: Callable[[], datetime],
        ttl: timedelta | None,
    ) -> None:
        """
        Args:
            url: The key. A document is identified by where it came from.
            http_cache: Persistent store, so politeness survives a restart.
            get_now: Injected clock.
            ttl: Lifetime from the host's policy. `None` is a declared `never`.
        """
        self._url = url
        self._http_cache = http_cache
        self._get_now = get_now
        self._ttl = ttl

    def get(self) -> HttpDocument | None:
        """The stored document while it is still fresh, otherwise None."""
        entry = self.stored()
        if entry is None or self._ttl is None:
            return None
        if self._get_now() - _as_aware(entry.fetched_at) >= self._ttl:
            return None
        return HttpDocument(
            body=entry.body, etag=entry.etag, last_modified=entry.last_modified
        )

    def put(self, value: HttpDocument) -> None:
        """Store the document, stamped from the injected clock."""
        if self._ttl is None:
            return
        self._http_cache.put(
            self._url,
            body=value.body,
            etag=value.etag,
            last_modified=value.last_modified,
            fetched_at=self._get_now(),
        )

    def stored(self) -> CachedResponse | None:
        """What is held for this URL regardless of age, for its validators.

        Deliberately separate from `get`. Staleness is a reason to *revalidate*,
        which needs the ETag of the copy we hold — a stale entry that forgot its
        validators would force a full download where a 304 would have done.

        A declared `never` returns nothing here either: it keeps nothing, and so
        it also gives up conditional requests. That makes `never` the wrong
        choice for a document fetch, and a considered one for a prompt.
        """
        if self._ttl is None:
            return None
        return self._http_cache.get(self._url)


class HttpFetcher:
    """One polite conditional GET, for every caller that speaks HTTP."""

    def __init__(
        self,
        *,
        session: requests.Session,
        network: NetworkConfig,
        policy: RequestPolicy,
        http_cache: HttpCache,
        get_now: Callable[[], datetime],
        logger: StructuredLogger | None = None,
    ) -> None:
        """
        Args:
            session: Injected HTTP session, so tests never reach the network.
                The transport is the boundary, so a structural fake belongs
                here — this is the one seam a double may legitimately stand in
                for.
            network: Read for the lifetime of what this caller stores. The key
                is the caller's; the lifetime is config's.
            policy: Throttle, retry and timeout.
            http_cache: Where bodies and validators are kept.
            get_now: Injected clock.
            logger: Structured logger. Optional.
        """
        self._session = session
        self._network = network
        self._policy = policy
        self._http_cache = http_cache
        self._get_now = get_now
        self._logger = logger
        self._is_transient = requests_transient_check(get_now=get_now)

    def get(
        self,
        url: str,
        *,
        label: str,
        params: QueryParams | None = None,
        cache_key: str | None = None,
        policy: str | None = None,
        max_age: timedelta | None = None,
    ) -> str:
        """Fetch a document, skipping the network whenever politeness allows.

        Args:
            url: Document to fetch. Its host decides the policy, unless one is
                named.
            label: Source name, for log messages.
            params: Query parameters. Part of what identifies the request, so
                they are part of the cache key by default — two addresses are
                two questions, not one answered twice.
            cache_key: What to store the answer under, when the URL and its
                query are the wrong key. Required whenever a parameter looks
                like a credential: the key is written to `http_cache`, so a
                token in the query string would be a secret at rest.
            policy: Names a policy directly, for a host that arrived from
                fetched data rather than from config.
            max_age: This document's own lifetime, overriding the category's.
                A feed's `min_fetch_interval_hours` is more specific than the
                policy covering ten scraped sites, and the specific one wins —
                including when it is *shorter*, which fetches more often than
                the category would. Zero means always revalidate, and still
                sends the validators, so the server can answer 304.

        Returns:
            The body, from cache when refetching would be impolite.

        Raises:
            ConfigError: If the host has no assigned policy.
            ValueError: If the cache key would carry a credential, or if the
                server answers 304 with nothing stored to serve.
        """
        key = _credential_free(
            cache_key if cache_key is not None else _keyed(url, params), label
        )
        host = _host_of(url)
        limits = (
            self._network.for_category(policy)
            if policy is not None
            else self._network.for_host(host)
        )
        cache = UrlCache(
            url=key,
            http_cache=self._http_cache,
            get_now=self._get_now,
            # A declared `never` is not overridable: it says this caller keeps
            # nothing, and a caller asking to reuse what was never kept gets a
            # fetch rather than a resurrection.
            ttl=limits.cache_ttl if max_age is None or limits.cache_ttl is None else max_age,
        )

        document = self._policy.call(
            host=host,
            perform=lambda timeout: self._perform(url, params, cache.stored(), timeout),
            is_transient=self._is_transient,
            cache=cache,
            label=label,
            policy=policy,
        )
        return document.body

    def _perform(
        self,
        url: str,
        params: QueryParams | None,
        stored: CachedResponse | None,
        timeout: float,
    ) -> HttpDocument:
        """One attempt: a conditional request, and what its answer means."""
        headers = {"User-Agent": USER_AGENT}
        if stored is not None:
            if stored.etag:
                headers["If-None-Match"] = stored.etag
            if stored.last_modified:
                headers["If-Modified-Since"] = stored.last_modified

        response = self._session.get(
            url, headers=headers, params=params, timeout=timeout
        )
        response.raise_for_status()

        if response.status_code == 304:
            if stored is None:
                raise ValueError(
                    f"{url} answered 304 with nothing stored to serve. Returning "
                    "the empty body would cache it as the truth."
                )
            return HttpDocument(
                body=stored.body,
                etag=stored.etag,
                last_modified=stored.last_modified,
            )

        return HttpDocument(
            body=response.text,
            etag=response.headers.get("ETag"),
            last_modified=response.headers.get("Last-Modified"),
        )


#: Query parameter names that carry a credential. The cache key is written to
#: the database, so a request whose key would contain one of these is refused
#: rather than silently storing a secret at rest. Structural: a caller that
#: means to cache such a request says what the key is.
_CREDENTIAL_PARAMS = frozenset(
    {"token", "api_key", "apikey", "key", "access_token", "auth", "password", "secret"}
)


def _credential_free(key: str, label: str) -> str:
    """The cache key, having confirmed it carries no credential we minted.

    `_keyed`'s two rules see only the derived path — a caller passing an
    explicit `cache_key` skips the function and both rules with it, and the
    hatch exists precisely for callers whose query carries a credential. This
    is the same property asked once, of the value that actually gets written,
    however it was arrived at. It also covers a credential sitting in the
    **URL**, which `_keyed` returns untouched when there are no params.

    It cannot name a parameter, because the key need not have come from one.
    So it names the source, which is what a reader needs and is safe to print:
    quoting the key would put the credential in a traceback, and a traceback
    is not routed through the log formatter that would have scrubbed it.

    Raises:
        ValueError: If the key carries a registered credential.
    """
    if contains_secret(key):
        raise ValueError(
            f"Refusing to store a cache key for '{label}': it carries a credential "
            "this process minted. The key is written to http_cache, so this would "
            "put a credential at rest. Key on what identifies the request, not on "
            "what authenticates it."
        )
    return key


def _keyed(url: str, params: QueryParams | None) -> str:
    """What identifies this request, for the cache.

    Sorted, so two callers spelling the same query in a different order share
    an answer rather than each storing their own.

    Two rules, and they are belt and braces rather than duplicates. The **name**
    rule catches a credential that was never minted as a `Secret` — a value the
    registry has never heard of. The **value** rule catches one that was, under
    a parameter name nobody would have put on a list, and it needs to know
    nothing about the provider to do it.

    Neither message may quote what it found. A `ValueError` reaches a traceback,
    and a traceback is not routed through the log formatter, so the scrub at
    that boundary would never see it.

    Raises:
        ValueError: If a parameter looks like a credential, or carries one, and
            no explicit `cache_key` was given.
    """
    if not params:
        return url

    offending = sorted(str(name) for name in params if str(name).lower() in _CREDENTIAL_PARAMS)
    if offending:
        raise ValueError(
            f"Refusing to build a cache key containing {', '.join(offending)}: the "
            "key is stored in http_cache, so this would put a credential at rest. "
            "Pass cache_key= naming what actually identifies the request."
        )

    minted = sorted(str(name) for name, value in params.items() if contains_secret(str(value)))
    if minted:
        raise ValueError(
            f"Refusing to build a cache key containing {', '.join(minted)}: its value "
            "is a credential this process minted, whatever the parameter is called. "
            "The key is stored in http_cache, so this would put a credential at rest. "
            "Pass cache_key= naming what actually identifies the request."
        )

    query = urlencode(sorted((str(k), str(v)) for k, v in params.items()))
    return f"{url}?{query}"


def _host_of(url: str) -> str:
    """The host a URL points at, which is what a policy is assigned to."""
    host = urlsplit(url).hostname
    if host is None:
        raise ValueError(f"Cannot tell which host {url!r} addresses")
    return host


def _as_aware(value: datetime) -> datetime:
    """Read a bare stamp as UTC, leaving one that states its zone alone.

    Our own writes are aware, but a row left by an older naive clock would
    otherwise raise on comparison — and since every source fetches through here,
    one legacy row would fail all of them at once.
    """
    return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)
