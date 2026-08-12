"""Unit tests for the polite fetch and its cache-age comparison."""

from __future__ import annotations

import io
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest

from src.ingestion.calendars.fetching import fetch_document
from src.storage.db import init_db
from src.storage.sqlite.http_cache import SqliteHttpCache
from src.utils.logging import get_logger

NOW = datetime(2026, 8, 8, 4, 0, tzinfo=timezone.utc)
URL = "https://example.org/feed.ics"


@pytest.fixture
def db(tmp_path):
    path = tmp_path / "test.db"
    init_db(path)
    return path


class _Session:
    def __init__(self, body: str = "FRESH") -> None:
        self.body = body
        self.requested: list[str] = []

    def get(self, url, headers=None, timeout=None):
        self.requested.append(url)
        response = MagicMock()
        response.status_code = 200
        response.text = self.body
        response.headers = {}
        response.raise_for_status.return_value = None
        return response


def _fetch(db, session, now=NOW, hours=6.0):
    return fetch_document(
        URL,
        session=session,
        db_path=db,
        get_now=lambda: now,
        min_fetch_interval_hours=hours,
        label="example",
        logger=get_logger("test", stream=io.StringIO()),
    )


def test_a_recent_cache_entry_is_reused(db):
    session = _Session()
    _fetch(db, session)

    body = _fetch(db, session, now=NOW + timedelta(hours=1))

    assert session.requested == [URL]
    assert body == "FRESH"


def test_an_expired_cache_entry_is_refetched(db):
    session = _Session()
    _fetch(db, session)

    _fetch(db, session, now=NOW + timedelta(hours=7))

    assert len(session.requested) == 2


def test_a_naive_cached_timestamp_does_not_break_the_fetch(db):
    """A cache row written by an older, naive clock must not kill every source.

    The batch clock is timezone-aware, so subtracting a stored naive timestamp
    raised `can't subtract offset-naive and offset-aware datetimes` — and since
    every configured source fetches through here, all seventeen failed at once.
    """
    SqliteHttpCache(db).put(
        URL,
        body="STALE",
        etag=None,
        last_modified=None,
        fetched_at=NOW.replace(tzinfo=None),
    )
    session = _Session()

    body = _fetch(db, session, now=NOW + timedelta(hours=1))

    assert body == "STALE"
    assert session.requested == []
