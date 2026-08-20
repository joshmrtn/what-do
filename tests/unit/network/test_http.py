"""Unit tests for the HTTP layer built on the request policy.

The policy inspects no URL and knows no transport. This is the one module that
is genuinely about HTTP: which statuses are worth another attempt, what a server
said with `Retry-After`, and the conditional request that lets it answer 304.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
import requests
from google.genai.errors import APIError

from src.config import ConfigError, NetworkConfig, NetworkPolicy
from src.network.http import (
    USER_AGENT,
    HttpDocument,
    HttpFetcher,
    UrlCache,
    api_status_transient_check,
    requests_transient_check,
)
from src.network.policy import RequestPolicy
from src.network.throttle import InMemoryThrottle
from src.storage.memory.http_cache import InMemoryHttpCache
from src.utils.secret import Secret

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
# An SDK error that carries the status it came from
# ---------------------------------------------------------------------------


def _api_error(status: int, headers: dict[str, str] | None = None) -> APIError:
    """What a vendor SDK raises for a failed HTTP call.

    The real `google.genai` error, deliberately, rather than a duck. The check
    reads `.code`, and a hand-rolled stand-in would still pass if the SDK spelled
    it something else — which is the only thing this check can get wrong.
    """
    response = requests.Response()
    response.status_code = status
    response.headers.update(headers or {})
    return APIError(status, {"error": {"message": "boom", "status": "X"}}, response)


@pytest.mark.parametrize("status", [429, 500, 502, 503, 504])
def test_a_server_side_sdk_failure_is_worth_retrying(status):
    """A status means the same thing whoever transported it."""
    assert api_status_transient_check(get_now=_clock())(_api_error(status)).retry


@pytest.mark.parametrize("status", [400, 401, 403, 404, 422])
def test_a_client_side_sdk_error_is_never_retried(status):
    assert not api_status_transient_check(get_now=_clock())(_api_error(status)).retry


def test_an_sdk_error_states_when_to_come_back():
    """The SDK keeps the response, so the server's own opinion survives it."""
    advice = api_status_transient_check(get_now=_clock())(
        _api_error(429, {"Retry-After": "30"})
    )
    assert advice.retry_after_seconds == pytest.approx(30.0)


def test_an_sdk_error_carrying_no_response_is_still_judged_by_its_status():
    """`response` is optional on an `APIError`; the code is not."""
    error = APIError(503, {"error": {"message": "unavailable", "status": "X"}})
    advice = api_status_transient_check(get_now=_clock())(error)
    assert advice.retry
    assert advice.retry_after_seconds is None


def test_an_error_carrying_no_status_is_not_retried():
    """Nothing to call transient, and a transport failure with no status is the
    caller's own check to add — this one only reads a status."""
    assert not api_status_transient_check(get_now=_clock())(TypeError("ours")).retry


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

    def get(self, url: str, *, headers: dict[str, str], params=None, timeout: float):
        self.calls.append(
            {"url": url, "headers": headers, "params": params, "timeout": timeout}
        )
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


def test_a_callers_headers_are_sent():
    """An Authorization line is why this parameter exists."""
    session = _FakeSession(_response())
    _fetcher(session).get(URL, label="x", headers={"Authorization": "Bearer abc"})
    assert session.calls[0]["headers"]["Authorization"] == "Bearer abc"


def test_a_caller_cannot_replace_the_politeness_headers():
    """The User-Agent identifying this project, and the validators that let a
    server answer 304, belong to the politeness contract rather than to
    whoever is asking. Ours win the collision."""
    session = _FakeSession(_response())
    _fetcher(session).get(URL, label="x", headers={"User-Agent": "Mozilla/5.0"})
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


# ---------------------------------------------------------------------------
# A caller that knows its document better than the category does
# ---------------------------------------------------------------------------


def test_a_shorter_max_age_refetches_before_the_policy_would():
    """`min_fetch_interval_hours` is per feed and the policy TTL is per
    category. The specific one wins, exactly as a venue's own name beats the
    label on the aggregator that listed it."""
    store = InMemoryHttpCache()
    session = _FakeSession(_response(body="v1"), _response(body="v2"))
    hour = timedelta(hours=1)
    _fetcher(session, http_cache=store, now=NOW).get(URL, label="x", max_age=hour)

    later = NOW + timedelta(hours=2)
    body = _fetcher(session, http_cache=store, now=later).get(URL, label="x", max_age=hour)

    assert body == "v2"
    assert len(session.calls) == 2


def test_a_longer_max_age_serves_past_the_policys_lifetime():
    store = InMemoryHttpCache()
    session = _FakeSession(_response(body="v1"))
    day = timedelta(days=1)
    _fetcher(session, http_cache=store, now=NOW).get(URL, label="x", max_age=day)

    later = NOW + timedelta(hours=12)
    body = _fetcher(session, http_cache=store, now=later).get(URL, label="x", max_age=day)

    assert body == "v1"
    assert len(session.calls) == 1


def test_no_max_age_leaves_the_policy_in_charge():
    store = InMemoryHttpCache()
    session = _FakeSession(_response(body="v1"), _response(body="v2"))
    _fetcher(session, http_cache=store, now=NOW).get(URL, label="x")

    later = NOW + timedelta(hours=7)
    _fetcher(session, http_cache=store, now=later).get(URL, label="x")

    assert len(session.calls) == 2


def test_a_max_age_of_zero_always_revalidates():
    """Zero disables reuse — the documented meaning of `min_fetch_interval_hours: 0`
    — and must not read as "no opinion, use the policy"."""
    store = InMemoryHttpCache()
    session = _FakeSession(_response(body="v1"), _response(body="v2"))
    zero = timedelta(0)
    _fetcher(session, http_cache=store, now=NOW).get(URL, label="x", max_age=zero)

    body = _fetcher(session, http_cache=store, now=NOW).get(URL, label="x", max_age=zero)

    assert body == "v2"
    assert len(session.calls) == 2


def test_a_max_age_cannot_revive_a_declared_never():
    """`never` says this caller stores nothing. A caller asking to reuse what
    was never kept gets a fetch, not a resurrection."""
    store = InMemoryHttpCache()
    session = _FakeSession(_response(body="v1"), _response(body="v2"))
    network = _test_network(cache_ttl=None)
    _fetcher(session, http_cache=store, now=NOW, network=network).get(
        URL, label="x", max_age=timedelta(days=1)
    )
    _fetcher(session, http_cache=store, now=NOW, network=network).get(
        URL, label="x", max_age=timedelta(days=1)
    )

    assert len(session.calls) == 2


def test_a_naive_stored_timestamp_does_not_break_the_fetch():
    """A cache row written by an older, naive clock must not kill every source.

    The batch clock is aware, so subtracting a stored naive timestamp raised
    `can't subtract offset-naive and offset-aware datetimes` — and since every
    configured source fetches through here, all seventeen failed at once.
    """
    store = InMemoryHttpCache()
    store.put(
        URL,
        body="STALE",
        etag=None,
        last_modified=None,
        fetched_at=NOW.replace(tzinfo=None),
    )
    session = _FakeSession()

    body = _fetcher(session, http_cache=store, now=NOW + timedelta(hours=1)).get(
        URL, label="example"
    )

    assert body == "STALE"
    assert session.calls == []


# ---------------------------------------------------------------------------
# Query parameters, and the credentials that hide in them
# ---------------------------------------------------------------------------


def test_params_are_sent_to_the_transport():
    session = _FakeSession(_response())
    _fetcher(session).get(URL, label="x", params={"q": "salem ma", "format": "json"})
    assert session.calls[0]["params"] == {"q": "salem ma", "format": "json"}


def test_params_are_part_of_what_identifies_the_request():
    """Two addresses are two questions, not one answered twice."""
    store = InMemoryHttpCache()
    session = _FakeSession(_response(body="salem"), _response(body="beverly"))
    _fetcher(session, http_cache=store).get(URL, label="x", params={"q": "salem"})
    body = _fetcher(session, http_cache=store).get(URL, label="x", params={"q": "beverly"})

    assert body == "beverly"
    assert len(session.calls) == 2


def test_the_same_question_is_asked_once():
    store = InMemoryHttpCache()
    session = _FakeSession(_response(body="salem"))
    _fetcher(session, http_cache=store).get(URL, label="x", params={"q": "salem"})
    body = _fetcher(session, http_cache=store).get(URL, label="x", params={"q": "salem"})

    assert body == "salem"
    assert len(session.calls) == 1


@pytest.mark.parametrize("secret", ["token", "api_key", "apikey", "key", "access_token"])
def test_a_credential_in_the_query_refuses_to_become_a_cache_key(secret):
    """The cache key is written to the database, so a token in the query string
    would be a credential at rest in `http_cache`. Structural, not remembered:
    a caller that means to cache says what the key is."""
    session = _FakeSession(_response())
    with pytest.raises(ValueError, match=secret):
        _fetcher(session).get(URL, label="x", params={secret: "s3cret", "q": "a"})
    assert session.calls == []


def test_an_explicit_cache_key_lets_a_credentialled_call_through():
    session = _FakeSession(_response(body="posts"))
    body = _fetcher(session).get(
        URL,
        label="x",
        params={"token": "s3cret", "usernames": "a,b"},
        cache_key=f"{URL}?usernames=a,b",
    )
    assert body == "posts"


def test_the_explicit_key_is_what_gets_stored():
    """And so the token never reaches the database."""
    store = InMemoryHttpCache()
    session = _FakeSession(_response(body="posts"))
    key = f"{URL}?usernames=a,b"
    _fetcher(session, http_cache=store).get(
        URL, label="x", params={"token": "s3cret", "usernames": "a,b"}, cache_key=key
    )

    assert store.get(key) is not None
    assert store.get(URL) is None


# ---------------------------------------------------------------------------
# The value check: a credential is refused whatever the parameter is called
# ---------------------------------------------------------------------------
#
# The name denylist above can only recognise the spellings we thought of, and a
# denylist is the thing this design replaces. Here the actual values are in
# hand, so the question is asked directly: is this value a credential we minted?
# Exact where the name rule guesses, and it needs to know nothing about the
# provider.
#
# The two rules are belt and braces, not duplicates. The name rule still catches
# a credential that was never minted as a `Secret` — a value the registry has
# never heard of — and the value rule catches one that was, under a name nobody
# would have put on a list.


#: Long enough to clear `MIN_REDACTABLE_LENGTH`, so minting registers it.
_MINTED = Secret("minted-credential-8b3f1c7e02")


@pytest.mark.parametrize("name", ["q", "sig", "usernames"])
def test_a_registered_credential_is_refused_from_the_query(name):
    """Innocuous names, and the value is what gives it away."""
    session = _FakeSession(_response())
    with pytest.raises(ValueError, match=name):
        _fetcher(session).get(URL, label="x", params={name: _MINTED.expose_secret()})
    assert session.calls == []


def test_an_explicit_cache_key_does_not_license_a_credential_in_the_query():
    """The hatch answers "what identifies this request", not "send anything".

    Naming a key keeps the credential out of `http_cache`, and that is all it
    does. The value still goes on the wire in the query string, where it reaches
    the provider's access log and any intermediary's — surfaces this process
    does not own and cannot scrub. Before the adapters moved to bearer headers
    this had to be allowed, because Apify's own call depended on it.
    """
    session = _FakeSession(_response())
    with pytest.raises(ValueError, match="sig"):
        _fetcher(session).get(
            URL,
            label="x",
            params={"sig": _MINTED.expose_secret()},
            cache_key=f"{URL}?identified-by=something-else",
        )
    assert session.calls == []


def test_the_query_refusal_points_at_the_header_form():
    """The advice must be the fix, and `cache_key=` is no longer the fix."""
    session = _FakeSession(_response())
    with pytest.raises(ValueError) as raised:
        _fetcher(session).get(URL, label="x", params={"sig": _MINTED.expose_secret()})

    assert "Authorization" in str(raised.value)


def test_the_refusal_names_the_parameter_and_never_the_value():
    """A message quoting the credential would be the leak it exists to stop.

    `ValueError` text reaches a traceback, and a traceback is not routed
    through the log formatter, so the scrub at that boundary would never see
    it. The one thing this must not do is print what it found.
    """
    session = _FakeSession(_response())
    with pytest.raises(ValueError) as raised:
        _fetcher(session).get(URL, label="x", params={"sig": _MINTED.expose_secret()})

    assert "sig" in str(raised.value)
    assert _MINTED.expose_secret() not in str(raised.value)


def test_a_credential_inside_a_longer_value_is_still_refused():
    """A signature built by concatenation carries the whole credential."""
    session = _FakeSession(_response())
    with pytest.raises(ValueError, match="sig"):
        _fetcher(session).get(
            URL, label="x", params={"sig": f"v1:{_MINTED.expose_secret()}:2026"}
        )


def test_an_explicit_cache_key_carrying_a_credential_is_refused():
    """The escape hatch is not an exemption.

    `cache_key=` exists so a caller whose query carries a credential can say
    what actually identifies the request — and it skips `_keyed` entirely, so
    both rules there are skipped with it. A caller who builds the key out of
    the credential anyway lands exactly where the hatch was meant to avoid.
    """
    session = _FakeSession(_response())
    with pytest.raises(ValueError, match="apify"):
        _fetcher(session).get(
            URL,
            label="apify",
            params={"usernames": "a,b"},
            cache_key=f"{URL}?token={_MINTED.expose_secret()}",
        )
    assert session.calls == []


def test_a_credential_in_the_url_itself_is_refused():
    """`_keyed` returns the URL untouched when there are no params."""
    session = _FakeSession(_response())
    with pytest.raises(ValueError, match="apify"):
        _fetcher(session).get(
            f"{URL}?token={_MINTED.expose_secret()}", label="apify"
        )
    assert session.calls == []


def test_the_key_refusal_names_the_label_and_never_the_value():
    """It cannot name a parameter — the key may not have come from one — so it
    names the source, which is what a reader needs and is safe to print."""
    session = _FakeSession(_response())
    with pytest.raises(ValueError) as raised:
        _fetcher(session).get(
            URL, label="apify", cache_key=f"{URL}?token={_MINTED.expose_secret()}"
        )

    assert "apify" in str(raised.value)
    assert _MINTED.expose_secret() not in str(raised.value)


def test_a_value_that_was_never_minted_is_keyed_unchanged():
    """Veezi's `siteToken`, which is the case this must not break.

    It is a public page identifier printed on the cinema's own website, so it
    is a plain `str` and was never registered — opaque and credential-shaped,
    and still not a credential. The type is what says so, and nobody has to
    remember it.
    """
    store = InMemoryHttpCache()
    session = _FakeSession(_response(body="showtimes"))
    site_token = "9f3a2c7e5b1d40886ec4"

    body = _fetcher(session, http_cache=store).get(
        URL, label="veezi", params={"siteToken": site_token}
    )

    assert body == "showtimes"
    assert store.get(f"{URL}?siteToken={site_token}") is not None
