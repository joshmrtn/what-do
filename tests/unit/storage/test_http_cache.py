"""Unit tests for the conditional-request HTTP cache."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from src.storage.sqlite.connection import init_db
from src.storage.http_cache import CachedResponse
from src.storage.sqlite.http_cache import SqliteHttpCache

FETCHED_AT = datetime(2026, 8, 5, 2, 0, tzinfo=timezone.utc)
URL = "https://example.com/calendar.ics"


@pytest.fixture
def db(tmp_path):
    path = tmp_path / "test.db"
    init_db(path)
    return path


def test_missing_url_reads_as_none(db):

    assert SqliteHttpCache(db).get(URL) is None


def test_round_trips_a_response(db):

    SqliteHttpCache(db).put(
        URL,
        body="BEGIN:VCALENDAR",
        etag='W/"abc"',
        last_modified="Wed, 05 Aug 2026 02:00:00 GMT",
        fetched_at=FETCHED_AT,
    )

    cached = SqliteHttpCache(db).get(URL)

    assert cached == CachedResponse(
        body="BEGIN:VCALENDAR",
        etag='W/"abc"',
        last_modified="Wed, 05 Aug 2026 02:00:00 GMT",
        fetched_at=FETCHED_AT,
    )


def test_validators_may_be_absent(db):
    """A server offering neither ETag nor Last-Modified still gets a cached body."""

    SqliteHttpCache(db).put(
        URL, body="X", etag=None, last_modified=None, fetched_at=FETCHED_AT)

    cached = SqliteHttpCache(db).get(URL)

    assert cached is not None
    assert cached.body == "X"
    assert cached.etag is None
    assert cached.last_modified is None


def test_writing_the_same_url_replaces_the_entry(db):

    SqliteHttpCache(db).put(
        URL, body="old", etag='"1"', last_modified=None, fetched_at=FETCHED_AT)
    SqliteHttpCache(db).put(
        URL, body="new", etag='"2"', last_modified=None, fetched_at=FETCHED_AT)

    cached = SqliteHttpCache(db).get(URL)

    assert cached is not None
    assert cached.body == "new"
    assert cached.etag == '"2"'


def test_entries_are_isolated_per_url(db):

    SqliteHttpCache(db).put(
        URL, body="one", etag=None, last_modified=None, fetched_at=FETCHED_AT)
    SqliteHttpCache(db).put(
        "https://other.example/c.ics", body="two",
        etag=None, last_modified=None, fetched_at=FETCHED_AT,
    )

    assert SqliteHttpCache(db).get(URL).body == "one"
    assert SqliteHttpCache(db).get("https://other.example/c.ics").body == "two"
