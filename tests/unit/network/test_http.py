"""Unit tests for the HTTP layer built on the request policy.

The policy inspects no URL and knows no transport. This is the one module that
is genuinely about HTTP: which statuses are worth another attempt, what a server
said with `Retry-After`, and the conditional request that lets it answer 304.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
import requests

from src.config import ConfigError, NetworkConfig, NetworkPolicy
from src.network.http import (
    USER_AGENT,
    HttpDocument,
    HttpFetcher,
    UrlCache,
    requests_transient_check,
)
from src.network.policy import RequestPolicy
from src.network.throttle import InMemoryThrottle
from src.storage.memory.http_cache import InMemoryHttpCache

NOW = datetime(2026, 8, 17, 12, 0, tzinfo=timezone.utc)
URL = "https://example.org/events"
DOCUMENT = HttpDocument(
    body="a listing page", etag='"abc"', last_modified="Wed, 01 Jul 2026 00:00:00 GMT"
)


def _clock(now: datetime = NOW):
    return lambda: now


def _http_error(status: int, headers: dict[str, str] | None = None) -> requests.HTTPError:
    """The error `raise_for_status` raises, with a response attached."""
    response = requests.Response()
    response.status_code = status
    response.headers.update(headers or {})
    return requests.HTTPError(f"{status} error", response=response)


# ---------------------------------------------------------------------------
# Which statuses are worth another attempt
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("status", [429, 500, 502, 503, 504])
def test_a_server_side_failure_is_worth_retrying(status):
    """429 and 5xx say "not now", which is different from "no"."""
    assert requests_transient_check(get_now=_clock())(_http_error(status)).retry


@pytest.mark.parametrize("status", [400, 401, 403, 404, 410, 422])
def test_a_client_error_is_never_retried(status):
    """A bad request repeated politely is still a bad request."""
    assert not requests_transient_check(get_now=_clock())(_http_error(status)).retry


def test_a_timeout_is_worth_retrying_on_our_own_schedule():
    check = requests_transient_check(get_now=_clock())
    advice = check(requests.Timeout("too slow"))
    assert advice.retry
    assert advice.retry_after_seconds is None


def test_a_connection_failure_is_worth_retrying():
    assert requests_transient_check(get_now=_clock())(
        requests.ConnectionError("refused")
    ).retry


def test_an_error_that_is_not_the_transports_is_not_retried():
    """A bug in our own code will fail identically next time."""
    assert not requests_transient_check(get_now=_clock())(TypeError("ours")).retry


def test_an_http_error_carrying_no_response_is_not_retried():
    """Without a status there is nothing to call transient, so do not guess."""
    assert not requests_transient_check(get_now=_clock())(
        requests.HTTPError("no response attached")
    ).retry


# ---------------------------------------------------------------------------
# Retry-After: the server named a number, so use its number
# ---------------------------------------------------------------------------


def test_retry_after_in_seconds_is_read_from_the_response():
    advice = requests_transient_check(get_now=_clock())(
        _http_error(429, {"Retry-After": "30"})
    )
    assert advice.retry_after_seconds == pytest.approx(30.0)


def test_retry_after_as_an_http_date_is_honoured_too():
    """A date form is as legal as delta-seconds, and ignoring it means hammering
    a server that asked for an hour."""
    when = NOW + timedelta(minutes=90)
    advice = requests_transient_check(get_now=_clock())(
        _http_error(503, {"Retry-After": when.strftime("%a, %d %b %Y %H:%M:%S GMT")})
    )
    assert advice.retry_after_seconds == pytest.approx(5400.0)


def test_a_retry_after_already_past_waits_no_negative_time():
    """A date behind us means "now", not a negative sleep."""
    when = NOW - timedelta(minutes=5)
    advice = requests_transient_check(get_now=_clock())(
        _http_error(503, {"Retry-After": when.strftime("%a, %d %b %Y %H:%M:%S GMT")})
    )
    assert advice.retry_after_seconds == pytest.approx(0.0)


def test_a_negative_delta_seconds_waits_no_negative_time():
    advice = requests_transient_check(get_now=_clock())(
        _http_error(429, {"Retry-After": "-10"})
    )
    assert advice.retry_after_seconds == pytest.approx(0.0)


def test_an_unreadable_retry_after_falls_back_to_our_own_backoff():
    """Nonsense in the header is no opinion, not a reason to abandon the retry."""
    advice = requests_transient_check(get_now=_clock())(
        _http_error(503, {"Retry-After": "soon-ish"})
    )
    assert advice.retry
    assert advice.retry_after_seconds is None


def test_retry_after_on_a_status_we_will_not_retry_is_still_not_retried():
    """The header does not turn a 404 into something worth asking twice."""
    advice = requests_transient_check(get_now=_clock())(
        _http_error(404, {"Retry-After": "30"})
    )
    assert not advice.retry


# ---------------------------------------------------------------------------
# The URL-keyed cache strategy
# ---------------------------------------------------------------------------


def _url_cache(
    *,
    http_cache: InMemoryHttpCache | None = None,
    ttl: timedelta | None = timedelta(hours=6),
    now: datetime = NOW,
) -> UrlCache:
    return UrlCache(
        url=URL,
        http_cache=http_cache if http_cache is not None else InMemoryHttpCache(),
        get_now=_clock(now),
        ttl=ttl,
    )


def test_nothing_stored_is_a_miss():
    assert _url_cache().get() is None


def test_what_was_stored_comes_back():
    store = InMemoryHttpCache()
    _url_cache(http_cache=store).put(DOCUMENT)
    assert _url_cache(http_cache=store).get() == DOCUMENT


def test_an_entry_inside_the_lifetime_is_served():
    store = InMemoryHttpCache()
    _url_cache(http_cache=store, now=NOW).put(DOCUMENT)

    later = NOW + timedelta(hours=6) - timedelta(seconds=1)
    assert _url_cache(http_cache=store, now=later).get() == DOCUMENT


def test_an_entry_past_the_lifetime_is_not_served():
    store = InMemoryHttpCache()
    _url_cache(http_cache=store, now=NOW).put(DOCUMENT)

    later = NOW + timedelta(hours=6, seconds=1)
    assert _url_cache(http_cache=store, now=later).get() is None


def test_the_stamp_comes_from_the_injected_clock():
    """`fetched_at` drives expiry, so it cannot come from the wall clock."""
    store = InMemoryHttpCache()
    _url_cache(http_cache=store, now=NOW).put(DOCUMENT)
    entry = store.get(URL)
    assert entry is not None and entry.fetched_at == NOW


def test_a_stale_entry_still_offers_its_validators():
    """Staleness is a reason to revalidate, not a reason to forget the ETag."""
    store = InMemoryHttpCache()
    _url_cache(http_cache=store, now=NOW).put(DOCUMENT)

    later = NOW + timedelta(days=30)
    cache = _url_cache(http_cache=store, now=later)
    assert cache.get() is None
    stored = cache.stored()
    assert stored is not None and stored.etag == DOCUMENT.etag


def test_a_declared_never_keeps_nothing_at_all():
    """`never` means this caller stores nothing — including validators, so it
    also gives up conditional requests. It is the wrong choice for a document."""
    store = InMemoryHttpCache()
    cache = _url_cache(http_cache=store, ttl=None)
    cache.put(DOCUMENT)

    assert cache.get() is None
    assert cache.stored() is None
    assert store.get(URL) is None


# ---------------------------------------------------------------------------
# The fetcher: one polite conditional GET
# ---------------------------------------------------------------------------


class _FakeSession:
    """Records requests and replays prepared responses.

    A fake at the transport boundary, which is where fakes belong. It makes no
    claim about our own behaviour — it hands back what a test told it to.
    """

    def __init__(self, *responses: requests.Response) -> None:
        self._responses = list(responses)
        self.calls: list[dict[str, object]] = []

    def get(self, url: str, *, headers: dict[str, str], timeout: float):
        self.calls.append({"url": url, "headers": headers, "timeout": timeout})
        if not self._responses:
            raise AssertionError(f"unexpected extra request to {url}")
        return self._responses.pop(0)


def _response(
    status: int = 200, body: str = "fresh body", headers: dict[str, str] | None = None
) -> requests.Response:
    response = requests.Response()
    response.status_code = status
    response._content = body.encode()
    response.headers.update(headers or {})
    return response


def _fetcher(
    session,
    *,
    http_cache: InMemoryHttpCache | None = None,
    now: datetime = NOW,
    network: NetworkConfig | None = None,
    sleeps: list[float] | None = None,
) -> HttpFetcher:
    clock = _clock(now)
    return HttpFetcher(
        session=session,
        network=network if network is not None else _test_network(),
        policy=RequestPolicy(
            network=network if network is not None else _test_network(),
            throttle=InMemoryThrottle(get_now=clock, sleep=lambda s: None),
            sleep=(sleeps.append if sleeps is not None else (lambda s: None)),
            random=lambda: 0.5,
        ),
        http_cache=http_cache if http_cache is not None else InMemoryHttpCache(),
        get_now=clock,
    )


def _test_network(cache_ttl: timedelta | None = timedelta(hours=6)) -> NetworkConfig:
    """One policy, assigned to the host in URL."""
    return NetworkConfig(
        policies={
            "listings": NetworkPolicy(
                min_interval_seconds=0.0,
                timeout_seconds=17.0,
                max_attempts=3,
                backoff_base_seconds=1.0,
                backoff_max_seconds=10.0,
                cache_ttl=cache_ttl,
            )
        },
        hosts={"example.org": "listings"},
    )


def test_the_body_is_returned():
    session = _FakeSession(_response(body="a listing page"))
    assert _fetcher(session).get(URL, label="example") == "a listing page"


def test_the_project_identifies_itself():
    """A User-Agent naming the project, rather than impersonating a browser."""
    session = _FakeSession(_response())
    _fetcher(session).get(URL, label="example")
    assert session.calls[0]["headers"]["User-Agent"] == USER_AGENT


def test_the_timeout_comes_from_the_host_policy():
    """Several callers pass none at all today, and none blocks for ever."""
    session = _FakeSession(_response())
    _fetcher(session).get(URL, label="example")
    assert session.calls[0]["timeout"] == pytest.approx(17.0)


def test_a_second_fetch_inside_the_lifetime_never_reaches_the_network():
    store = InMemoryHttpCache()
    session = _FakeSession(_response(body="once"))
    _fetcher(session, http_cache=store, now=NOW).get(URL, label="example")

    later = NOW + timedelta(hours=1)
    body = _fetcher(session, http_cache=store, now=later).get(URL, label="example")

    assert body == "once"
    assert len(session.calls) == 1


def test_past_the_lifetime_it_revalidates_with_what_the_server_gave_it():
    store = InMemoryHttpCache()
    first = _response(
        body="v1", headers={"ETag": '"abc"', "Last-Modified": "Wed, 01 Jul 2026 00:00:00 GMT"}
    )
    session = _FakeSession(first, _response(body="v2"))
    _fetcher(session, http_cache=store, now=NOW).get(URL, label="example")

    later = NOW + timedelta(hours=7)
    body = _fetcher(session, http_cache=store, now=later).get(URL, label="example")

    assert body == "v2"
    assert session.calls[1]["headers"]["If-None-Match"] == '"abc"'
    assert (
        session.calls[1]["headers"]["If-Modified-Since"]
        == "Wed, 01 Jul 2026 00:00:00 GMT"
    )


def test_a_304_serves_the_stored_body():
    """The politest outcome short of not asking: the server sends no body."""
    store = InMemoryHttpCache()
    session = _FakeSession(
        _response(body="unchanged", headers={"ETag": '"abc"'}), _response(status=304, body="")
    )
    _fetcher(session, http_cache=store, now=NOW).get(URL, label="example")

    later = NOW + timedelta(hours=7)
    body = _fetcher(session, http_cache=store, now=later).get(URL, label="example")
    assert body == "unchanged"


def test_a_304_refreshes_the_stamp_so_the_interval_starts_again():
    store = InMemoryHttpCache()
    session = _FakeSession(
        _response(body="unchanged", headers={"ETag": '"abc"'}), _response(status=304, body="")
    )
    _fetcher(session, http_cache=store, now=NOW).get(URL, label="example")

    later = NOW + timedelta(hours=7)
    _fetcher(session, http_cache=store, now=later).get(URL, label="example")

    entry = store.get(URL)
    assert entry is not None and entry.fetched_at == later


def test_a_304_with_nothing_stored_is_refused():
    """Returning `response.text` here would cache an empty document as the
    truth. There is nothing to serve, so this is an error, not a body."""
    session = _FakeSession(_response(status=304, body=""))
    with pytest.raises(ValueError, match="304"):
        _fetcher(session).get(URL, label="example")


def test_a_transient_failure_is_retried_up_to_the_policys_attempts():
    sleeps: list[float] = []
    session = _FakeSession(_response(status=503), _response(status=503), _response(body="third"))
    body = _fetcher(session, sleeps=sleeps).get(URL, label="example")

    assert body == "third"
    assert len(session.calls) == 3
    assert len(sleeps) == 2


def test_a_client_error_is_not_retried():
    session = _FakeSession(_response(status=404))
    with pytest.raises(requests.HTTPError):
        _fetcher(session).get(URL, label="example")
    assert len(session.calls) == 1


def test_a_host_with_no_declared_policy_is_refused_by_name():
    session = _FakeSession(_response())
    with pytest.raises(ConfigError, match="elsewhere.test"):
        _fetcher(session).get("https://elsewhere.test/feed", label="stranger")
    assert session.calls == []


def test_a_policy_may_be_named_for_a_host_that_came_from_data():
    """An image URL points at whatever CDN a venue uses, so those hosts cannot
    be listed in config. Naming the policy keeps it a decision."""
    session = _FakeSession(_response(body="an image"))
    body = _fetcher(session).get(
        "https://cdn.unknowable.test/x.jpg", label="image", policy="listings"
    )
    assert body == "an image"
