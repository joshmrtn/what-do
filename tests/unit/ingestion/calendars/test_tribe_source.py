"""Unit tests for TribeCalendarSource.

The Events Calendar caps its iCal export at 30 events. `paged` and
`tribe_paged` do nothing; `tribe-bar-date` moves the window, so the export is
walked by date rather than by page.
"""

from __future__ import annotations

import io
from datetime import datetime, time, timezone
from unittest.mock import MagicMock

import pytest

from src.storage.memory.http_cache import InMemoryHttpCache
from src.config import FeedConfig
from src.ingestion.calendars.tribe_source import TribeCalendarSource
from src.storage.sqlite.connection import init_db
from src.utils.logging import get_logger

FIXED_NOW = datetime(2026, 8, 7, 6, 0, tzinfo=timezone.utc)
URL = "https://example.org/events/?ical=1"


@pytest.fixture
def cache():
    """No database: these tests are about conditional requests."""
    return InMemoryHttpCache()


def _vevent(uid: str, day: str) -> str:
    return (
        f"BEGIN:VEVENT\r\nUID:{uid}\r\nSUMMARY:Event {uid}\r\n"
        f"DTSTART:{day}T230000Z\r\nEND:VEVENT\r\n"
    )


def _calendar(*events: str) -> str:
    return "BEGIN:VCALENDAR\r\nVERSION:2.0\r\n" + "".join(events) + "END:VCALENDAR\r\n"


class _Windows:
    """Serves a different calendar per `tribe-bar-date`, counting requests."""

    def __init__(self, by_bar_date: dict[str | None, str]) -> None:
        self._by_bar_date = by_bar_date
        self.requested: list[str | None] = []

    def get(self, url, headers=None, timeout=None):
        bar = None
        if "tribe-bar-date=" in url:
            bar = url.split("tribe-bar-date=")[1].split("&")[0]
        self.requested.append(bar)
        response = MagicMock()
        response.status_code = 200
        response.text = self._by_bar_date.get(bar, _calendar())
        response.headers = {}
        response.raise_for_status.return_value = None
        return response


def _make_source(cache, windows, max_requests=8, horizon_days=45, export_cap=2):
    """`export_cap` defaults to 2 so small fixtures still look full and walk."""
    http = _Windows(windows)
    source = TribeCalendarSource(
        config=FeedConfig(name="tribe", url=URL, source_type="tribe"),
        http_cache=cache,
        session=http,
        get_now=lambda: FIXED_NOW,
        logger=get_logger("test", stream=io.StringIO()),
        timezone_name="America/New_York",
        horizon_days=horizon_days,
        day_starts_at=time(4, 0),
        max_requests=max_requests,
        export_cap=export_cap,
    )
    return source, http


class TestWalking:
    def test_a_window_short_of_the_cap_is_the_last_one(self, cache):
        """Fewer events than the export holds means there is no more to fetch."""
        source, http = _make_source(
            cache, {None: _calendar(_vevent("a", "20260810"))}, export_cap=30
        )

        assert len(source.fetch()) == 1
        assert http.requested == [None]

    def test_the_walk_continues_from_the_last_date_seen(self, cache):
        windows = {
            None: _calendar(_vevent("a", "20260810"), _vevent("b", "20260820")),
            "2026-08-20": _calendar(_vevent("b", "20260820"), _vevent("c", "20260830")),
        }
        source, http = _make_source(cache, windows)

        results = source.fetch()

        assert len(results) == 3
        assert http.requested == [None, "2026-08-20", "2026-08-30"]

    def test_an_event_seen_twice_is_emitted_once(self, cache):
        """Windows overlap by design — the boundary date is refetched so the
        walk cannot step over an event that shares it."""
        windows = {
            None: _calendar(_vevent("a", "20260810"), _vevent("b", "20260820")),
            "2026-08-20": _calendar(_vevent("b", "20260820")),
        }
        source, _ = _make_source(cache, windows)

        assert len({c.id for c in source.fetch()}) == 2

    def test_the_walk_stops_when_nothing_new_arrives(self, cache):
        repeated = _calendar(_vevent("a", "20260810"), _vevent("b", "20260812"))
        source, http = _make_source(cache, {None: repeated, "2026-08-12": repeated})

        source.fetch()

        assert http.requested == [None, "2026-08-12"]

    def test_the_walk_stops_at_an_empty_window(self, cache):
        windows = {None: _calendar(_vevent("a", "20260810"), _vevent("b", "20260812"))}
        source, http = _make_source(cache, windows)

        source.fetch()

        assert http.requested == [None, "2026-08-12"]

    def test_the_walk_stops_past_the_horizon(self, cache):
        """Nothing beyond the horizon is ranked, so nothing beyond it is worth asking for."""
        windows = {
            None: _calendar(_vevent("a", "20260810"), _vevent("b", "20261225")),
            "2026-12-25": _calendar(_vevent("c", "20270101")),
        }
        source, http = _make_source(cache, windows, horizon_days=45)

        source.fetch()

        assert http.requested == [None]

    def test_the_walk_never_exceeds_its_request_cap(self, cache):
        """A calendar that always reports a later date must not become a crawl."""
        # Every window is full and names a later date, so only the cap stops it.
        windows = {None: _calendar(_vevent("a", "20260808"), _vevent("b", "20260809"))}
        for i in range(9, 28):
            windows[f"2026-08-{i:02d}"] = _calendar(
                _vevent(f"e{i}", f"202608{i:02d}"), _vevent(f"f{i}", f"202608{i + 1:02d}")
            )
        source, http = _make_source(cache, windows, max_requests=3)

        source.fetch()

        assert len(http.requested) == 3


class TestMapping:
    def test_candidates_carry_the_feeds_identity(self, cache):
        source, _ = _make_source(cache, {None: _calendar(_vevent("a", "20260810"))})

        candidate = source.fetch()[0]

        assert candidate.source == "tribe"
        assert candidate.id == "tribe:a"
        assert candidate.raw_published_at is None
