"""Contract every HttpCache implementation must satisfy.

The cache exists so politeness survives a process restart: the batch runs
unattended, but a person debugging it may run it several times in an evening,
and without persisted validators every run is a full download of somebody
else's server.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from src.storage.db import init_db
from src.storage.memory.http_cache import InMemoryHttpCache
from src.storage.sqlite.http_cache import SqliteHttpCache

_NOW = datetime(2026, 8, 11, 12, 0, tzinfo=timezone.utc)
_URL = "https://example.test/feed.ics"


@pytest.fixture(params=["sqlite", "memory"])
def cache(request, tmp_path):
    if request.param == "sqlite":
        path = tmp_path / "http.db"
        init_db(path)
        return SqliteHttpCache(path)
    return InMemoryHttpCache()


def _put(cache, url=_URL, body="BEGIN:VCALENDAR", etag='W/"abc"',
         last_modified="Mon, 11 Aug 2026 00:00:00 GMT", fetched_at=_NOW):
    cache.put(url, body=body, etag=etag, last_modified=last_modified,
              fetched_at=fetched_at)


class TestRoundTrip:
    def test_a_stored_response_reads_back_whole(self, cache):
        _put(cache)

        stored = cache.get(_URL)

        assert stored.body == "BEGIN:VCALENDAR"
        assert stored.etag == 'W/"abc"'
        assert stored.last_modified == "Mon, 11 Aug 2026 00:00:00 GMT"
        assert stored.fetched_at == _NOW

    def test_a_url_never_fetched_reads_back_as_nothing(self, cache):
        assert cache.get(_URL) is None

    def test_a_server_offering_no_validators_still_caches(self, cache):
        """Plenty of sites send neither, and the politeness floor still applies."""
        _put(cache, etag=None, last_modified=None)

        stored = cache.get(_URL)
        assert stored.body == "BEGIN:VCALENDAR"
        assert (stored.etag, stored.last_modified) == (None, None)

    def test_refetching_replaces_the_earlier_entry(self, cache):
        _put(cache, body="old", etag='W/"old"')
        _put(cache, body="new", etag='W/"new"')

        stored = cache.get(_URL)
        assert (stored.body, stored.etag) == ("new", 'W/"new"')

    def test_each_url_is_its_own_entry(self, cache):
        _put(cache, url="https://a.test/f", body="a")
        _put(cache, url="https://b.test/f", body="b")

        assert cache.get("https://a.test/f").body == "a"
        assert cache.get("https://b.test/f").body == "b"


class TestRevalidation:
    def test_a_304_can_restamp_without_losing_the_body(self, cache):
        """A 304 means unchanged: keep the body, move the clock on."""
        _put(cache, body="original", fetched_at=_NOW)
        later = datetime(2026, 8, 11, 18, 0, tzinfo=timezone.utc)

        stored = cache.get(_URL)
        cache.put(_URL, body=stored.body, etag=stored.etag,
                  last_modified=stored.last_modified, fetched_at=later)

        refreshed = cache.get(_URL)
        assert refreshed.body == "original"
        assert refreshed.fetched_at == later
