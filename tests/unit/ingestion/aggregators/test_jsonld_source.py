"""Unit tests for JsonLdEventSource."""

from __future__ import annotations

import io
from datetime import datetime, time, timedelta, timezone
from unittest.mock import MagicMock

import pytest

from src.config import FeedConfig
from src.ingestion.aggregators.jsonld_source import JsonLdEventSource
from src.models.event_candidate import EventCandidate
from src.models.timing import EXACT
from src.storage.sqlite.connection import init_db
from src.utils.logging import get_logger
from tests.support.network import fetcher_for

#: 02:00 in New York, so the night in progress began the previous day.
FIXED_NOW = datetime(2026, 8, 8, 6, 0, tzinfo=timezone.utc)
URL = "https://www.pem.org/events"
EASTERN = timezone(timedelta(hours=-4))


@pytest.fixture
def db(tmp_path):
    path = tmp_path / "test.db"
    init_db(path)
    return path


def _event(
    name: str = "Drop-in Art Making",
    start: str = "2026-08-08T11:00:00-04:00",
    end: str | None = "2026-08-08T15:00:00-04:00",
    url: str = "https://www.pem.org/events/drop-in",
    venue: str = "Peabody Essex Museum",
) -> str:
    end_part = f', "endDate": "{end}"' if end else ""
    return (
        f'{{"@type": "Event", "name": "{name}", "startDate": "{start}"{end_part}, '
        f'"url": "{url}", "description": "A description", '
        f'"location": {{"@type": "Place", "name": "{venue}", '
        f'"address": {{"@type": "PostalAddress", '
        f'"streetAddress": "161 Essex Street, Salem, MA 01970"}}}}}}'
    )


def _page(*events: str) -> str:
    return (
        '<html><body><script type="application/ld+json">['
        + ", ".join(events)
        + "]</script></body></html>"
    )


class _FakeSession:
    def __init__(self, body: str) -> None:
        self._body = body
        self.requested: list[str] = []

    def get(self, url, headers=None, params=None, timeout=None):
        self.requested.append(url)
        response = MagicMock()
        response.status_code = 200
        response.text = self._body
        response.headers = {}
        response.raise_for_status.return_value = None
        return response


def _make_source(db, body, horizon_days=45, **overrides):
    settings = {"name": "pem", "url": URL, "source_type": "pem"}
    settings.update(overrides)
    http = _FakeSession(body)
    source = JsonLdEventSource(
        config=FeedConfig(**settings),
        fetcher=fetcher_for(http, urls=URL, now=FIXED_NOW),
        get_now=lambda: FIXED_NOW,
        logger=get_logger("test", stream=io.StringIO()),
        timezone_name="America/New_York",
        horizon_days=horizon_days,
        day_starts_at=time(4, 0),
    )
    return source, http


class TestMapping:
    def test_returns_event_candidates(self, db):
        source, _ = _make_source(db, _page(_event()))

        results = source.fetch()

        assert len(results) == 1
        assert isinstance(results[0], EventCandidate)

    def test_the_id_identifies_an_occurrence_not_a_programme(self, db):
        """A recurring programme keeps one URL across every date it runs."""
        source, _ = _make_source(db, _page(_event(url="https://www.pem.org/events/tube-bugs")))

        assert source.fetch()[0].id == (
            "pem:https://www.pem.org/events/tube-bugs@2026-08-08T11:00:00-04:00"
        )

    def test_two_dates_of_one_programme_are_two_candidates(self, db):
        body = _page(
            _event(name="Weekly drop-in", start="2026-08-08T11:00:00-04:00", url="https://x/y"),
            _event(name="Weekly drop-in", start="2026-08-15T11:00:00-04:00", url="https://x/y"),
        )
        source, _ = _make_source(db, body)

        ids = [c.id for c in source.fetch()]
        assert len(set(ids)) == 2

    def test_an_event_without_a_url_still_gets_a_stable_id(self, db):
        """Identity has to survive a source that omits the field."""
        body = _page(_event().replace('"url": "https://www.pem.org/events/drop-in", ', ""))
        source, _ = _make_source(db, body)

        first = source.fetch()[0].id
        source, _ = _make_source(db, body)
        assert source.fetch()[0].id == first

    def test_the_start_keeps_the_offset_the_source_stated(self, db):
        source, _ = _make_source(db, _page(_event()))

        assert source.fetch()[0].start_time == datetime(2026, 8, 8, 11, 0, tzinfo=EASTERN)

    def test_timing_is_always_exact(self, db):
        source, _ = _make_source(db, _page(_event()))

        assert source.fetch()[0].timing == EXACT

    def test_the_venue_and_city_come_from_the_markup(self, db):
        source, _ = _make_source(db, _page(_event()))

        candidate = source.fetch()[0]
        assert candidate.venue == "Peabody Essex Museum"
        assert candidate.location == "161 Essex Street, Salem, MA 01970"

    def test_the_configured_venue_fills_in_when_the_markup_names_none(self, db):
        body = _page(_event().replace(
            ', "location": {"@type": "Place", "name": "Peabody Essex Museum", '
            '"address": {"@type": "PostalAddress", '
            '"streetAddress": "161 Essex Street, Salem, MA 01970"}}', ""
        ))
        source, _ = _make_source(db, body, venue="PEM", city="Salem")

        candidate = source.fetch()[0]
        assert candidate.venue == "PEM"

    def test_no_announcement_date_is_invented(self, db):
        source, _ = _make_source(db, _page(_event()))

        assert source.fetch()[0].raw_published_at is None


class TestWindow:
    def test_an_event_past_the_horizon_is_dropped(self, db):
        body = _page(_event(), _event(name="Far", start="2026-12-01T19:00:00-05:00"))
        source, _ = _make_source(db, body, horizon_days=45)

        assert [c.title for c in source.fetch()] == ["Drop-in Art Making"]

    def test_an_event_before_the_night_floor_is_dropped(self, db):
        body = _page(_event(name="Old", start="2026-07-01T19:00:00-04:00"), _event())
        source, _ = _make_source(db, body)

        assert [c.title for c in source.fetch()] == ["Drop-in Art Making"]

    def test_an_event_earlier_tonight_is_kept(self, db):
        body = _page(_event(name="Tonight", start="2026-08-07T20:00:00-04:00", end=None))
        source, _ = _make_source(db, body)

        assert [c.title for c in source.fetch()] == ["Tonight"]


class TestFetching:
    def test_costs_one_request(self, db):
        source, http = _make_source(db, _page(_event()))

        source.fetch()

        assert http.requested == [URL]

    def test_a_page_with_no_structured_data_yields_nothing(self, db):
        source, _ = _make_source(db, "<html><body>Nothing</body></html>")

        assert source.fetch() == []
