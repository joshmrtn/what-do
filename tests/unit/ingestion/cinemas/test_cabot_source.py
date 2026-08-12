"""Unit tests for CabotListingSource."""

from __future__ import annotations

import io
from datetime import datetime, time, timedelta, timezone
from unittest.mock import MagicMock

import pytest

from src.storage.memory.http_cache import InMemoryHttpCache
from src.config import FeedConfig
from src.ingestion.cinemas.cabot_source import CabotListingSource
from src.models.event_candidate import EventCandidate
from src.storage.db import init_db
from src.utils.logging import get_logger

FIXED_NOW = datetime(2026, 8, 7, 6, 0, tzinfo=timezone.utc)  # 02:00 in New York
URL = "https://thecabot.org/whats-on/"


@pytest.fixture
def db(tmp_path):
    path = tmp_path / "test.db"
    init_db(path)
    return path


def _item(event_id: str, day: int, month: str = "Aug", when: str = "8:00pm",
          title: str = "A Show", subtitle: str = "", off_site: bool = False) -> str:
    sub = f'<p class="h5">{subtitle}</p>' if subtitle else ""
    marker = '<img class="off_cabot_logo" alt="Off Cabot Event">' if off_site else ""
    return f"""
    <div class="event_item" id="event_item_{event_id}"><div class="event_item_inner">
      <div class="event_thumb">{marker}<a href="https://thecabot.org/event/e{event_id}/"></a></div>
      <div class="event_info">
        <div class="event_date"><span>{day}</span> {month}
          <div class="time">{when}</div></div>
        <div class="event_text"><div class="genre">Music</div>
          <p class="h4">{title}</p>{sub}</div>
      </div></div></div>"""


def _page(*items: str, total: int = 10) -> str:
    return (
        '<html><body><div class="events_holder">'
        f'<p class="results_count">Showing 1-10 of {total} events</p>'
        f'{"".join(items)}</div></body></html>'
    )


class _Pages:
    """Serves a different body per page URL, counting the requests made."""

    def __init__(self, bodies: dict[str, str]) -> None:
        self._bodies = bodies
        self.requested: list[str] = []

    def get(self, url, headers=None, timeout=None):
        self.requested.append(url)
        response = MagicMock()
        response.status_code = 200
        response.text = self._bodies.get(url, _page(total=0))
        response.headers = {}
        response.raise_for_status.return_value = None
        return response


def _make_source(db, bodies, horizon_days=45, max_pages=12, **overrides):
    settings = {
        "name": "cabot",
        "url": URL,
        "source_type": "cabot",
        "venue": "The Cabot",
        "city": "Beverly",
    }
    settings.update(overrides)
    http = _Pages(bodies)
    source = CabotListingSource(
        config=FeedConfig(**settings),
        http_cache=InMemoryHttpCache(),
        session=http,
        get_now=lambda: FIXED_NOW,
        logger=get_logger("test", stream=io.StringIO()),
        timezone_name="America/New_York",
        horizon_days=horizon_days,
        day_starts_at=time(4, 0),
        max_pages=max_pages,
    )
    return source, http


class TestMapping:
    def test_returns_event_candidates(self, db):
        source, _ = _make_source(db, {URL: _page(_item("1", 7), total=1)})

        results = source.fetch()

        assert len(results) == 1
        assert isinstance(results[0], EventCandidate)

    def test_the_id_comes_from_the_site_and_is_stable(self, db):
        source, _ = _make_source(db, {URL: _page(_item("20591", 7), total=1)})

        assert source.fetch()[0].id == "cabot:20591"

    def test_the_venue_comes_from_config(self, db):
        source, _ = _make_source(db, {URL: _page(_item("1", 7), total=1)})

        assert source.fetch()[0].venue == "The Cabot"

    def test_an_off_site_event_takes_its_venue_from_the_subtitle(self, db):
        """Several events are at Off Cabot rather than the theatre itself."""
        body = _page(
            _item("1", 7, subtitle="Off Cabot - 9 Wallis St, Beverly", off_site=True),
            total=1,
        )
        source, _ = _make_source(db, {URL: body})

        assert source.fetch()[0].venue == "Off Cabot"

    def test_an_off_site_address_is_kept_in_the_description(self, db):
        """Off Cabot is a different building a few streets from the theatre."""
        body = _page(
            _item("1", 7, subtitle="Off Cabot - 9 Wallis St, Beverly", off_site=True),
            total=1,
        )
        source, _ = _make_source(db, {URL: body})

        candidate = source.fetch()[0]

        assert candidate.venue == "Off Cabot"
        assert "9 Wallis St, Beverly" in candidate.description

    def test_a_tour_name_is_not_mistaken_for_a_venue(self, db):
        """The subtitle is overloaded; only the off-site marker tells which it is."""
        body = _page(_item("1", 7, subtitle="INDIGO PARK TOUR"), total=1)
        source, _ = _make_source(db, {URL: body})

        candidate = source.fetch()[0]

        assert candidate.venue == "The Cabot"
        assert "INDIGO PARK TOUR" in candidate.description

    def test_genres_become_the_description(self, db):
        """Extraction cannot infer 'Comedy' from 'In Focus: Improv Lab'."""
        source, _ = _make_source(db, {URL: _page(_item("1", 7), total=1)})

        assert source.fetch()[0].description == "Music"

    def test_the_start_is_aware_in_the_venues_zone(self, db):
        source, _ = _make_source(db, {URL: _page(_item("1", 7), total=1)})

        candidate = source.fetch()[0]

        assert candidate.start_time.hour == 20
        assert candidate.start_time.utcoffset() == timedelta(hours=-4)

    def test_no_candidate_claims_a_published_date(self, db):
        source, _ = _make_source(db, {URL: _page(_item("1", 7), total=1)})

        assert source.fetch()[0].raw_published_at is None


class TestPagination:
    def test_a_second_page_is_followed(self, db):
        bodies = {
            URL: _page(*[_item(str(i), 7 + i) for i in range(10)], total=12),
            f"{URL.rstrip('/')}/page/2/": _page(_item("99", 20), total=12),
        }
        source, http = _make_source(db, bodies)

        results = source.fetch()

        assert len(results) == 11
        assert len(http.requested) == 2

    def test_paging_stops_once_a_page_opens_past_the_horizon(self, db):
        """The listing is ascending, so nothing later can be closer."""
        bodies = {
            URL: _page(*[_item(str(i), 7 + i) for i in range(10)], total=40),
            f"{URL.rstrip('/')}/page/2/": _page(_item("99", 1, month="Dec"), total=40),
        }
        source, http = _make_source(db, bodies, horizon_days=45)

        source.fetch()

        assert len(http.requested) == 2

    def test_paging_never_exceeds_the_cap(self, db):
        """A pagination bug on their server must not become a crawl on ours."""
        bodies = {URL: _page(*[_item(str(i), 7) for i in range(10)], total=9999)}
        for page in range(2, 20):
            bodies[f"{URL.rstrip('/')}/page/{page}/"] = _page(
                *[_item(f"p{page}n{i}", 7) for i in range(10)], total=9999
            )
        source, http = _make_source(db, bodies, max_pages=3)

        source.fetch()

        assert len(http.requested) == 3

    def test_a_single_page_listing_asks_for_nothing_more(self, db):
        source, http = _make_source(db, {URL: _page(_item("1", 7), total=1)})

        source.fetch()

        assert len(http.requested) == 1

    def test_an_empty_page_stops_paging(self, db):
        bodies = {URL: _page(*[_item(str(i), 7 + i) for i in range(10)], total=50)}
        source, http = _make_source(db, bodies)

        source.fetch()

        assert len(http.requested) == 2


class TestWindow:
    def test_an_event_past_the_horizon_is_not_emitted(self, db):
        body = _page(_item("1", 7), _item("2", 1, month="Dec"), total=2)
        source, _ = _make_source(db, {URL: body}, horizon_days=45)

        assert [c.id for c in source.fetch()] == ["cabot:1"]


class TestRepeatedPages:
    """Many sites serve page 1 for any out-of-range page number."""

    def test_a_page_repeating_earlier_events_stops_the_walk(self, db):
        first = _page(*[_item(str(i), 7 + i) for i in range(10)], total=9999)
        bodies = {URL: first}
        for page in range(2, 20):
            bodies[f"{URL.rstrip('/')}/page/{page}/"] = first
        source, http = _make_source(db, bodies, max_pages=12)

        results = source.fetch()

        assert len(results) == 10
        assert len(http.requested) == 2
