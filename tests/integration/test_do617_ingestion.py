"""Do617 parsing against real captured venue pages.

No network: the fixtures are real responses from Do617 venue pages, captured on
2026-08-07 with scripts and styles emptied. They are the only place the parser
meets the site's actual chrome — navigation, footer and sidebar all carry
`itemprop` and `name` attributes of their own.
"""

import io
from datetime import datetime, time, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.config import FeedConfig
from src.ingestion.aggregators.do617_listing import parse_do617
from src.ingestion.aggregators.do617_source import Do617VenueSource
from src.storage.sqlite.connection import init_db
from src.utils.logging import get_logger
from tests.support.network import fetcher_for

FIXTURES = Path(__file__).parent.parent / "fixtures"

EASTERN = timezone(timedelta(hours=-4))

VENUE_URL = "https://do617.com/venues/gulu-gulu-cafe"

#: The pages were captured on this date, and their events run forward from it.
FIXED_NOW = datetime(2026, 8, 7, 10, 0, tzinfo=timezone.utc)


def _fixture(name: str) -> str:
    return (FIXTURES / name).read_text()


@pytest.fixture
def db(tmp_path):
    path = tmp_path / "test.db"
    init_db(path)
    return path


class _Pages:
    def __init__(self, bodies: dict[str, str]) -> None:
        self._bodies = bodies
        self.requested: list[str] = []

    def get(self, url, headers=None, params=None, timeout=None):
        self.requested.append(url)
        response = MagicMock()
        response.status_code = 200
        response.text = self._bodies[url]
        response.headers = {}
        response.raise_for_status.return_value = None
        return response


class TestGuluGulu:
    def test_reads_every_event_on_the_page(self) -> None:
        page = parse_do617(_fixture("do617_gulu_gulu.html"))

        assert len(page.events) == 25

    def test_reads_the_first_event_whole(self) -> None:
        page = parse_do617(_fixture("do617_gulu_gulu.html"))

        event = page.events[0]
        assert event.title == "Eva James - Live Music No Cover"
        assert event.start == datetime(2026, 8, 7, 20, 0, tzinfo=EASTERN)
        assert event.end == datetime(2026, 8, 7, 23, 0, tzinfo=EASTERN)
        assert event.permalink == "/events/2026/8/7/eva-james-live-music-no-cover-tickets"
        assert event.venue == "Gulu-Gulu Cafe"
        assert event.venue_slug == "gulu-gulu-cafe"
        assert event.street == "247 Essex St"
        assert event.city == "Salem"
        assert event.region == "MA"
        assert event.latitude == 42.5650452

    def test_events_are_ascending_and_within_the_listed_range(self) -> None:
        page = parse_do617(_fixture("do617_gulu_gulu.html"))

        starts = [event.start for event in page.events]
        assert starts == sorted(starts)
        assert starts[-1] == datetime(2026, 9, 2, 18, 0, tzinfo=EASTERN)

    def test_every_event_has_a_distinct_permalink(self) -> None:
        page = parse_do617(_fixture("do617_gulu_gulu.html"))

        permalinks = [event.permalink for event in page.events]
        assert len(set(permalinks)) == len(permalinks)

    def test_site_chrome_never_becomes_an_event(self) -> None:
        page = parse_do617(_fixture("do617_gulu_gulu.html"))

        assert all(event.venue == "Gulu-Gulu Cafe" for event in page.events)
        assert all(event.permalink.startswith("/events/") for event in page.events)

    def test_finds_the_next_page(self) -> None:
        page = parse_do617(_fixture("do617_gulu_gulu.html"))

        assert page.next_page_url == "/venues/gulu-gulu-cafe?page=2"


class TestSecondPage:
    def test_continues_where_the_first_page_stopped(self) -> None:
        first = parse_do617(_fixture("do617_gulu_gulu.html"))
        second = parse_do617(_fixture("do617_gulu_gulu_page2.html"))

        assert len(second.events) == 19
        assert second.events[0].start > first.events[-1].start

    def test_shares_no_events_with_the_first_page(self) -> None:
        first = {event.permalink for event in parse_do617(_fixture("do617_gulu_gulu.html")).events}
        second = {
            event.permalink for event in parse_do617(_fixture("do617_gulu_gulu_page2.html")).events
        }

        assert first & second == set()

    def test_the_last_page_offers_no_next(self) -> None:
        page = parse_do617(_fixture("do617_gulu_gulu_page2.html"))

        assert page.next_page_url is None


class TestEmptyVenue:
    def test_a_venue_with_no_listings_yields_no_events(self) -> None:
        # Koto's page is valid and carries the venue's address, but Do617 has
        # nothing upcoming for it. That is a normal state, not a failure.
        page = parse_do617(_fixture("do617_koto.html"))

        assert page.events == []
        assert page.next_page_url is None


def _source(db, bodies, horizon_days=45, url=VENUE_URL, name="do617_gulu_gulu"):
    http = _Pages(bodies)
    source = Do617VenueSource(
        config=FeedConfig(name=name, url=url, source_type="do617"),
        fetcher=fetcher_for(
            http,
            urls=VENUE_URL,
            now=FIXED_NOW,
        ),
        get_now=lambda: FIXED_NOW,
        logger=get_logger("test", stream=io.StringIO()),
        timezone_name="America/New_York",
        horizon_days=horizon_days,
        day_starts_at=time(4, 0),
    )
    return source, http


class TestWalkingTheRealVenue:
    def test_walks_both_pages_and_stops(self, db):
        bodies = {
            VENUE_URL: _fixture("do617_gulu_gulu.html"),
            f"{VENUE_URL}?page=2": _fixture("do617_gulu_gulu_page2.html"),
        }
        source, http = _source(db, bodies, horizon_days=120)

        candidates = source.fetch()

        assert http.requested == [VENUE_URL, f"{VENUE_URL}?page=2"]
        assert len(candidates) == 44

    def test_a_short_horizon_stops_the_walk_on_the_first_page(self, db):
        """Page 2 opens on 3 September, past a 14-day horizon from the capture."""
        bodies = {VENUE_URL: _fixture("do617_gulu_gulu.html")}
        source, http = _source(db, bodies, horizon_days=14)

        candidates = source.fetch()

        assert http.requested == [VENUE_URL]
        assert all(c.start_time < datetime(2026, 8, 21, 4, 0, tzinfo=EASTERN) for c in candidates)

    def test_candidates_carry_what_the_markup_stated(self, db):
        bodies = {VENUE_URL: _fixture("do617_gulu_gulu.html")}
        source, _ = _source(db, bodies, horizon_days=14)

        first = source.fetch()[0]
        assert first.id == "do617_gulu_gulu:/events/2026/8/7/eva-james-live-music-no-cover-tickets"
        assert first.title == "Eva James - Live Music No Cover"
        assert first.venue == "Gulu-Gulu Cafe"
        assert first.location == "Salem"
        assert first.start_time == datetime(2026, 8, 7, 20, 0, tzinfo=EASTERN)
        assert first.url == (
            "https://do617.com/events/2026/8/7/eva-james-live-music-no-cover-tickets"
        )

    def test_ids_are_unique_across_the_whole_walk(self, db):
        bodies = {
            VENUE_URL: _fixture("do617_gulu_gulu.html"),
            f"{VENUE_URL}?page=2": _fixture("do617_gulu_gulu_page2.html"),
        }
        source, _ = _source(db, bodies, horizon_days=120)

        ids = [c.id for c in source.fetch()]
        assert len(set(ids)) == len(ids)

    def test_an_empty_venue_costs_one_request_and_yields_nothing(self, db):
        url = "https://do617.com/venues/koto"
        source, http = _source(db, {url: _fixture("do617_koto.html")}, url=url, name="do617_koto")

        assert source.fetch() == []
        assert http.requested == [url]
