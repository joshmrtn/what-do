"""Unit tests for the TMDb provider going through the policy.

TMDb has never had a cache. It is also the one provider where a **miss is the
common answer** — a cinema listing carries titles TMDb does not recognise — and
a miss that is not cached is a request that repeats for ever. That is the same
failure the forecast horizon fixed, in a different costume.

A cached miss cannot be `None`: `RequestPolicy.call` reads `None` from a cache
strategy as "nothing stored" and calls anyway. So the cached value is a
`MovieLookup`, which is an object whether or not it found anything.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from urllib.parse import urlencode

import pytest
import requests

from src.enrichment.movies import TMDB_HOST, TMDbProvider, title_key
from src.storage.memory.movie_cache import InMemoryMovieCache
from src.utils.secret import Secret
from tests.support.network import fetcher_policy

NOW = datetime(2026, 8, 17, 12, 0, tzinfo=timezone.utc)
URL = f"https://{TMDB_HOST}/3"

_SEARCH = {"results": [{"id": 42, "release_date": "1985-07-03"}]}
_DETAIL = {
    "genres": [{"name": "Adventure"}, {"name": "Comedy"}],
    "runtime": 116,
    "overview": "A teenager is sent back to 1955.",
    "release_date": "1985-07-03",
}


def _response(payload: dict, status: int = 200) -> requests.Response:
    response = requests.Response()
    response.status_code = status
    response._content = json.dumps(payload).encode()
    return response


class _FakeSession:
    def __init__(self, *responses: requests.Response) -> None:
        self._responses = list(responses)
        self.calls: list[dict] = []

    def get(self, url: str, *, params=None, headers=None, timeout=None):
        self.calls.append(
            {"url": url, "params": params, "headers": headers, "timeout": timeout}
        )
        if not self._responses:
            raise AssertionError(f"unexpected extra request to {url}")
        return self._responses.pop(0)


def _found_session() -> _FakeSession:
    return _FakeSession(_response(_SEARCH), _response(_DETAIL))


def _provider(
    session,
    *,
    cache=None,
    now: datetime = NOW,
    cache_ttl: timedelta | None = timedelta(days=7),
) -> TMDbProvider:
    return TMDbProvider(
        read_access_token=Secret("a-key"),
        session=session,
        policy=fetcher_policy(urls=URL, now=now),
        movie_cache=cache if cache is not None else InMemoryMovieCache(),
        cache_ttl=cache_ttl,
        get_now=lambda: now,
    )


# ---------------------------------------------------------------------------
# The lookup
# ---------------------------------------------------------------------------


def test_metadata_comes_back():
    data = _provider(_found_session()).fetch("Back to the Future", 1985)
    assert data == {
        "genres": ["Adventure", "Comedy"],
        "runtime_minutes": 116,
        "summary": "A teenager is sent back to 1955.",
        "release_year": 1985,
    }


def test_the_timeout_comes_from_the_policy():
    session = _found_session()
    _provider(session).fetch("Back to the Future", 1985)
    assert session.calls[0]["timeout"] == pytest.approx(30.0)


def test_a_title_tmdb_does_not_know_is_no_metadata():
    session = _FakeSession(_response({"results": []}))
    assert _provider(session).fetch("A Local Short Film", None) is None


def test_a_transient_failure_is_retried():
    session = _FakeSession(_response({}, status=503), _response(_SEARCH), _response(_DETAIL))
    assert _provider(session).fetch("Back to the Future", 1985) is not None


def test_a_programming_error_is_not_reported_as_no_metadata():
    """A bug must not be able to look like a film TMDb has never heard of."""
    session = _FakeSession()
    session.get = lambda *a, **k: (_ for _ in ()).throw(NameError("oops"))
    with pytest.raises(NameError):
        _provider(session).fetch("Back to the Future", 1985)


# ---------------------------------------------------------------------------
# The cache, including the misses
# ---------------------------------------------------------------------------


def test_a_second_lookup_of_one_film_makes_no_request():
    store = InMemoryMovieCache()
    _provider(_found_session(), cache=store).fetch("Back to the Future", 1985)

    session = _FakeSession()  # any request at all raises
    assert _provider(session, cache=store).fetch("Back to the Future", 1985) is not None


def test_a_miss_is_cached_so_it_is_not_asked_again():
    """The forecast lesson restated: nothing caches a `None`, so 74 dates were
    re-requested every run. A title TMDb does not know is the same shape."""
    store = InMemoryMovieCache()
    _provider(_FakeSession(_response({"results": []})), cache=store).fetch("Unknown", None)

    session = _FakeSession()  # any request at all raises
    assert _provider(session, cache=store).fetch("Unknown", None) is None


def test_a_stored_answer_past_its_lifetime_is_asked_again():
    """Seven days: a film's runtime does not move, but TMDb fills in a summary
    or a poster for a new release days after it first appears."""
    store = InMemoryMovieCache()
    _provider(_found_session(), cache=store, now=NOW).fetch("Back to the Future", 1985)

    later = NOW + timedelta(days=7, seconds=1)
    session = _found_session()
    _provider(session, cache=store, now=later).fetch("Back to the Future", 1985)

    assert len(session.calls) == 2


def test_the_same_film_at_a_different_year_is_a_different_question():
    store = InMemoryMovieCache()
    _provider(_found_session(), cache=store).fetch("The Thing", 1982)

    session = _found_session()
    _provider(session, cache=store).fetch("The Thing", 2011)

    assert len(session.calls) == 2


def test_a_declared_never_caches_nothing():
    store = InMemoryMovieCache()
    _provider(_found_session(), cache=store, cache_ttl=None).fetch("BTTF", 1985)

    session = _found_session()
    _provider(session, cache=store, cache_ttl=None).fetch("BTTF", 1985)

    assert len(session.calls) == 2


# ---------------------------------------------------------------------------
# The key
# ---------------------------------------------------------------------------


def test_case_and_spacing_do_not_make_a_second_question():
    """Compare on a canonical key; the stored title is still what was asked."""
    assert title_key("  Back  to   the FUTURE ") == title_key("back to the future")


def test_one_cinemas_spelling_hits_the_others_cached_answer():
    store = InMemoryMovieCache()
    _provider(_found_session(), cache=store).fetch("Back to the Future", 1985)

    session = _FakeSession()  # any request at all raises
    assert _provider(session, cache=store).fetch("BACK TO THE FUTURE", 1985) is not None


def test_different_films_are_not_collapsed():
    assert title_key("The Thing") != title_key("The Thing II")


# ---------------------------------------------------------------------------
# How the credential travels
# ---------------------------------------------------------------------------
#
# TMDb's v3 API key can *only* go in the query string. The v4 read access token
# goes in an `Authorization` header, so moving to it is what makes the header
# form available at all — which is why v3 support is removed outright rather
# than kept as a fallback. Keeping it would mean keeping a credential in a URL
# and depending on the log scrubbing to hide it: a backstop standing in for a
# fix. It costs nothing, because TMDb has never been switched on.


def test_both_requests_send_the_bearer_token():
    """Search and detail are one question, and both must authenticate."""
    session = _found_session()
    _provider(session).fetch("Back to the Future", 1985)

    assert [c["headers"]["Authorization"] for c in session.calls] == [
        "Bearer a-key",
        "Bearer a-key",
    ]


def test_no_request_carries_a_credential_in_the_query():
    session = _found_session()
    _provider(session).fetch("Back to the Future", 1985)

    for call in session.calls:
        assert "api_key" not in (call["params"] or {})
        assert "a-key" not in urlencode(call["params"] or {})


def test_the_search_query_is_otherwise_unchanged():
    """Only the credential moves; the question itself is the same one."""
    session = _found_session()
    _provider(session).fetch("Back to the Future", 1985)

    assert session.calls[0]["params"] == {"query": "Back to the Future", "year": 1985}
