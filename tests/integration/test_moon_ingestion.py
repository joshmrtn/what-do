"""MOON ingestion against the real captured feed.

No network: the fixture is MOON's `/shows?format=rss` as returned on 2026-08-07.

Every item in it is dated June 2026, because MOON has paused new bookings during
a City of Salem Building Department dispute and is relocating scheduled shows.
The feed is adopted anyway, so shows appear the night booking resumes. Two of
these tests pin what that means today: run against the capture's own June, the
feed yields shows; run against August, it yields nothing at all.
"""

import io
from datetime import datetime, time, timezone
from pathlib import Path
from unittest.mock import MagicMock
from zoneinfo import ZoneInfo

import pytest

from src.config import FeedConfig
from src.ingestion.calendars.moon_source import MoonRssSource
from src.models.timing import EXACT, UNKNOWN
from src.storage.db import init_db
from src.utils.logging import get_logger

FIXTURE = Path(__file__).parent.parent / "fixtures" / "moon_shows.rss"
URL = "https://www.moon-ns.org/shows?format=rss"
EASTERN = ZoneInfo("America/New_York")

#: Inside the run of shows the captured feed describes.
JUNE = datetime(2026, 6, 1, 10, 0, tzinfo=timezone.utc)

#: The day the feed was captured, by which every show in it has passed.
AUGUST = datetime(2026, 8, 7, 10, 0, tzinfo=timezone.utc)


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


def _source(db, now):
    http = _FakeSession(FIXTURE.read_text())
    source = MoonRssSource(
        config=FeedConfig(name="moon", url=URL, source_type="moon"),
        db_path=db,
        session=http,
        get_now=lambda: now,
        logger=get_logger("test", stream=io.StringIO()),
        timezone_name="America/New_York",
        horizon_days=45,
        day_starts_at=time(4, 0),
    )
    return source, http


class TestTheLiveFeed:
    def test_yields_the_shows_still_ahead(self, db):
        source, _ = _source(db, JUNE)

        candidates = source.fetch()

        # Six of the twenty items are dated on or after 1 June and not cancelled.
        assert len(candidates) == 6

    def test_reads_a_templated_item_whole(self, db):
        source, _ = _source(db, JUNE)

        show = next(c for c in source.fetch() if "Viraya" in c.title)
        assert show.title == "Viraya, Missed Opportunities, Abandon All Closure at Felt Fanatic"
        assert show.start_time == datetime(2026, 6, 27, 18, 0, tzinfo=EASTERN)
        assert show.timing == EXACT
        assert show.venue == "Felt Fanatic"
        assert show.url.startswith("https://www.moon-ns.org/shows/")

    def test_an_item_with_no_published_hour_says_so(self, db):
        source, _ = _source(db, JUNE)

        show = next(c for c in source.fetch() if "Cherubhead" in c.title)
        assert show.timing == UNKNOWN
        assert show.start_time == datetime(2026, 6, 25, 4, 0, tzinfo=EASTERN)

    def test_the_cancelled_show_never_becomes_a_candidate(self, db):
        source, _ = _source(db, JUNE)

        assert not any("NAGLY" in c.title for c in source.fetch())

    def test_no_start_is_ever_the_publication_date(self, db):
        """Every item was posted in May. None of them happens in May."""
        source, _ = _source(db, JUNE)

        assert all(c.start_time.month != 5 for c in source.fetch())

    def test_announcement_dates_are_kept(self, db):
        source, _ = _source(db, JUNE)

        assert all(c.raw_published_at is not None for c in source.fetch())

    def test_every_candidate_has_a_distinct_id(self, db):
        source, _ = _source(db, JUNE)

        ids = [c.id for c in source.fetch()]
        assert len(set(ids)) == len(ids)


class TestTheDormantFeed:
    def test_a_feed_of_past_shows_yields_nothing(self, db):
        """MOON is paused. Configured now so shows appear when booking resumes."""
        source, http = _source(db, AUGUST)

        assert source.fetch() == []
        assert http.requested == [URL]
