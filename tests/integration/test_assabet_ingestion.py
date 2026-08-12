"""Salem Public Library ingestion against the real captured feed.

No network: the fixture is `salempl.assabetinteractive.com/calendar/
upcoming-events.rss` as returned on 2026-08-08.

The library publishes no feed on its own domain. This one was found by following
an `<iframe>` embed on `salempl.org/calendar/` to the Assabet host — the same
trick that found Cape Ann's calendar behind a Google Calendar embed.
"""

import io
from datetime import datetime, time, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.storage.memory.http_cache import InMemoryHttpCache
from src.config import FeedConfig
from src.ingestion.calendars.assabet_source import AssabetRssSource
from src.models.timing import EXACT
from src.storage.sqlite.connection import init_db
from src.utils.logging import get_logger

FIXTURE = Path(__file__).parent.parent / "fixtures" / "salempl_upcoming.rss"
URL = "https://salempl.assabetinteractive.com/calendar/upcoming-events.rss"
EASTERN = timezone(timedelta(hours=-4))

#: The morning the feed was captured; its listings run forward from it.
FIXED_NOW = datetime(2026, 8, 8, 10, 0, tzinfo=timezone.utc)


class _FakeSession:
    def __init__(self, body: str) -> None:
        self._body = body
        self.requested: list[str] = []

    def get(self, url, headers=None, timeout=None):
        self.requested.append(url)
        response = MagicMock()
        response.status_code = 200
        response.text = self._body
        response.headers = {}
        response.raise_for_status.return_value = None
        return response


@pytest.fixture
def db(tmp_path):
    path = tmp_path / "test.db"
    init_db(path)
    return path


def _source(db, horizon_days=45):
    http = _FakeSession(FIXTURE.read_text())
    source = AssabetRssSource(
        config=FeedConfig(name="salempl", url=URL, source_type="salempl", city="Salem"),
        http_cache=InMemoryHttpCache(),
        session=http,
        get_now=lambda: FIXED_NOW,
        logger=get_logger("test", stream=io.StringIO()),
        timezone_name="America/New_York",
        horizon_days=horizon_days,
        day_starts_at=time(4, 0),
    )
    return source, http


class TestTheLiveFeed:
    def test_one_request_yields_the_library_programme(self, db):
        source, http = _source(db)

        candidates = source.fetch()

        assert http.requested == [URL]
        assert len(candidates) == 27

    def test_every_event_has_a_stated_hour(self, db):
        """pubDate carries the time here, so nothing is ever placed."""
        source, _ = _source(db)

        assert all(c.timing == EXACT for c in source.fetch())

    def test_reads_an_in_library_event_whole(self, db):
        source, _ = _source(db)

        event = next(c for c in source.fetch() if c.title == "Family Circle Time")
        assert event.start_time == datetime(2026, 8, 8, 10, 30, tzinfo=EASTERN)
        assert event.venue == "The Salem Public Library"
        assert event.location == "Salem"
        assert "sensory-friendly" in event.description

    def test_an_off_site_event_names_where_it_actually_is(self, db):
        source, _ = _source(db)

        event = next(c for c in source.fetch() if c.title == "Farmers' Market Storytime")
        assert event.venue == "Salem Farmers' Market"

    def test_no_description_still_carries_the_preamble(self, db):
        """The date and address are structured fields, not part of what it is."""
        source, _ = _source(db)

        for candidate in source.fetch():
            assert candidate.description is None or "Salem, MA, 01970" not in candidate.description

    def test_starts_are_ascending_and_inside_the_window(self, db):
        source, _ = _source(db)

        starts = [c.start_time for c in source.fetch()]
        assert starts == sorted(starts)
        assert starts[0] >= datetime(2026, 8, 7, 4, 0, tzinfo=EASTERN)

    def test_every_candidate_has_a_distinct_id(self, db):
        source, _ = _source(db)

        ids = [c.id for c in source.fetch()]
        assert len(set(ids)) == len(ids)

    def test_a_short_horizon_leaves_the_later_events_behind(self, db):
        source, _ = _source(db, horizon_days=7)

        assert len(source.fetch()) < 27
