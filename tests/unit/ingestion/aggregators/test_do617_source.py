"""Unit tests for Do617VenueSource."""

from __future__ import annotations

import io
from datetime import datetime, time, timedelta, timezone
from unittest.mock import MagicMock

import pytest

from src.config import FeedConfig
from src.ingestion.aggregators.do617_source import Do617VenueSource
from src.models.event_candidate import EventCandidate
from src.models.timing import EXACT
from src.storage.sqlite.connection import init_db
from src.utils.logging import get_logger
from tests.support.network import fetcher_for

#: 02:00 in New York, so the night in progress began the previous day.
FIXED_NOW = datetime(2026, 8, 7, 6, 0, tzinfo=timezone.utc)
URL = "https://do617.com/venues/gulu-gulu-cafe"


@pytest.fixture
def db(tmp_path):
    path = tmp_path / "test.db"
    init_db(path)
    return path


def _card(
    permalink: str,
    start: str,
    end: str | None = None,
    title: str = "A Show",
    venue: str | None = "Gulu-Gulu Cafe",
    city: str | None = "Salem",
    category: str = "music",
) -> str:
    end_html = f'<meta itemprop="endDate" datetime="{end}" content="{end}"/>' if end else ""
    venue_html = (
        f'<a href="/venues/gulu-gulu-cafe" itemprop="url">'
        f'<span itemprop="name">{venue}</span></a>'
        if venue
        else ""
    )
    city_html = f'<meta itemprop="addressLocality" content="{city}" />' if city else ""
    return f"""
    <div class="ds-listing event-card ds-event-category-{category}"
         data-permalink="{permalink}" itemprop="event" itemscope
         itemtype="http://schema.org/Event">
      <a href="{permalink}" itemprop="url">
        <span class="ds-listing-event-title-text" itemprop="name">{title}</span>
      </a>
      <div class="ds-listing-details">
        <div itemprop="location" itemscope itemtype="http://schema.org/Place">
          {venue_html}
          <span itemprop="address" itemscope itemtype="http://schema.org/PostalAddress">
            <meta itemprop="streetAddress" content="247 Essex St" />
            {city_html}
          </span>
        </div>
        <meta itemprop="startDate" datetime="{start}" content="{start}"/>
        {end_html}
      </div>
    </div>"""


def _page(*cards: str, next_page: str | None = None) -> str:
    nav = f'<a href="{next_page}" class="ds-next-page" rel="next">Next</a>' if next_page else ""
    return f'<html><body><div class="ds-listings">{"".join(cards)}</div>{nav}</body></html>'


class _Pages:
    """Serves a body per page URL, counting the requests made."""

    def __init__(self, bodies: dict[str, str]) -> None:
        self._bodies = bodies
        self.requested: list[str] = []

    def get(self, url, headers=None, timeout=None):
        self.requested.append(url)
        response = MagicMock()
        response.status_code = 200
        response.text = self._bodies.get(url, _page())
        response.headers = {}
        response.raise_for_status.return_value = None
        return response


def _make_source(db, bodies, horizon_days=45, max_pages=6, **overrides):
    settings = {"name": "do617_gulu_gulu", "url": URL, "source_type": "do617"}
    settings.update(overrides)
    http = _Pages(bodies)
    source = Do617VenueSource(
        config=FeedConfig(**settings),
        fetcher=fetcher_for(http, urls=URL, now=FIXED_NOW),
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
        source, _ = _make_source(db, {URL: _page(_card("/events/2026/8/7/a", "2026-08-07T20:00-0400"))})

        results = source.fetch()

        assert len(results) == 1
        assert isinstance(results[0], EventCandidate)

    def test_the_id_comes_from_the_sites_own_permalink(self, db):
        source, _ = _make_source(
            db, {URL: _page(_card("/events/2026/8/7/eva-james", "2026-08-07T20:00-0400"))}
        )

        assert source.fetch()[0].id == "do617_gulu_gulu:/events/2026/8/7/eva-james"

    def test_the_start_is_the_instant_the_source_stated(self, db):
        """Do617 states an offset, so the instant is unambiguous. The candidate
        holds it in UTC — one representation, because `for_window` compares
        stored timestamps as text and text only sorts like time at a fixed
        offset. Equality is instant equality, so the source's own form is the
        honest way to write the expectation."""
        source, _ = _make_source(db, {URL: _page(_card("/events/2026/8/7/a", "2026-08-07T20:00-0400"))})

        start = source.fetch()[0].start_time
        assert start == datetime(2026, 8, 7, 20, 0, tzinfo=timezone(timedelta(hours=-4)))
        assert start.utcoffset() == timedelta(0)

    def test_timing_is_always_exact(self, db):
        """Unlike every other listing, Do617 states the hour and the offset."""
        source, _ = _make_source(db, {URL: _page(_card("/events/2026/8/7/a", "2026-08-07T20:00-0400"))})

        assert source.fetch()[0].timing == EXACT

    def test_the_end_time_is_carried_when_given(self, db):
        source, _ = _make_source(
            db,
            {URL: _page(_card("/events/2026/8/7/a", "2026-08-07T20:00-0400", end="2026-08-07T23:00-0400"))},
        )

        assert source.fetch()[0].end_time == datetime(
            2026, 8, 7, 23, 0, tzinfo=timezone(timedelta(hours=-4))
        )

    def test_the_venue_comes_from_the_card_not_the_config(self, db):
        """An aggregator lists many venues, so the feed cannot declare one."""
        source, _ = _make_source(
            db,
            {URL: _page(_card("/events/2026/8/7/a", "2026-08-07T20:00-0400", venue="Koto"))},
        )

        assert source.fetch()[0].venue == "Koto"

    def test_the_configured_venue_fills_in_when_the_card_names_none(self, db):
        source, _ = _make_source(
            db,
            {URL: _page(_card("/events/2026/8/7/a", "2026-08-07T20:00-0400", venue=None))},
            venue="Gulu-Gulu Cafe",
        )

        assert source.fetch()[0].venue == "Gulu-Gulu Cafe"

    def test_the_city_comes_from_the_cards_address(self, db):
        source, _ = _make_source(db, {URL: _page(_card("/events/2026/8/7/a", "2026-08-07T20:00-0400"))})

        assert source.fetch()[0].location == "Salem"

    def test_the_url_is_absolute(self, db):
        source, _ = _make_source(
            db, {URL: _page(_card("/events/2026/8/7/eva-james", "2026-08-07T20:00-0400"))}
        )

        assert source.fetch()[0].url == "https://do617.com/events/2026/8/7/eva-james"

    def test_the_category_reaches_extraction_as_a_description(self, db):
        """`ds-event-category-drag` is the site's own label for what this is."""
        source, _ = _make_source(
            db,
            {URL: _page(_card("/events/2026/8/7/a", "2026-08-07T20:00-0400", category="food-drink"))},
        )

        assert source.fetch()[0].description == "food & drink"

    def test_no_announcement_date_is_invented(self, db):
        """A listing carries none, and that field is what the lookback discards on."""
        source, _ = _make_source(db, {URL: _page(_card("/events/2026/8/7/a", "2026-08-07T20:00-0400"))})

        assert source.fetch()[0].raw_published_at is None


class TestWindow:
    def test_an_event_past_the_horizon_is_dropped(self, db):
        body = _page(
            _card("/events/2026/8/7/a", "2026-08-07T20:00-0400"),
            _card("/events/2026/12/1/b", "2026-12-01T20:00-0500"),
        )
        source, _ = _make_source(db, {URL: body}, horizon_days=45)

        assert [c.id for c in source.fetch()] == ["do617_gulu_gulu:/events/2026/8/7/a"]

    def test_an_event_before_the_night_floor_is_dropped(self, db):
        """The venue page still lists shows from earlier in the run's own night."""
        body = _page(
            _card("/events/2026/8/1/old", "2026-08-01T20:00-0400"),
            _card("/events/2026/8/7/a", "2026-08-07T20:00-0400"),
        )
        source, _ = _make_source(db, {URL: body})

        assert [c.id for c in source.fetch()] == ["do617_gulu_gulu:/events/2026/8/7/a"]

    def test_an_event_earlier_tonight_is_kept(self, db):
        """Ingestion floors at the night, not the instant it happens to run."""
        body = _page(_card("/events/2026/8/6/tonight", "2026-08-06T20:00-0400"))
        source, _ = _make_source(db, {URL: body})

        assert [c.id for c in source.fetch()] == ["do617_gulu_gulu:/events/2026/8/6/tonight"]


class TestPagination:
    def test_a_page_without_a_next_link_ends_the_walk(self, db):
        source, http = _make_source(db, {URL: _page(_card("/events/2026/8/7/a", "2026-08-07T20:00-0400"))})

        source.fetch()

        assert http.requested == [URL]

    def test_follows_the_next_link(self, db):
        bodies = {
            URL: _page(
                _card("/events/2026/8/7/a", "2026-08-07T20:00-0400"),
                next_page="/venues/gulu-gulu-cafe?page=2",
            ),
            f"{URL}?page=2": _page(_card("/events/2026/8/9/b", "2026-08-09T20:00-0400")),
        }
        source, http = _make_source(db, bodies)

        results = source.fetch()

        assert http.requested == [URL, f"{URL}?page=2"]
        assert [c.id for c in results] == [
            "do617_gulu_gulu:/events/2026/8/7/a",
            "do617_gulu_gulu:/events/2026/8/9/b",
        ]

    def test_a_page_opening_past_the_horizon_ends_the_walk(self, db):
        """Ascending order means every later page is past it too."""
        bodies = {
            URL: _page(
                _card("/events/2026/12/1/far", "2026-12-01T20:00-0500"),
                next_page="/venues/gulu-gulu-cafe?page=2",
            ),
            f"{URL}?page=2": _page(_card("/events/2027/1/1/further", "2027-01-01T20:00-0500")),
        }
        source, http = _make_source(db, bodies)

        assert source.fetch() == []
        assert http.requested == [URL]

    def test_a_page_straddling_the_horizon_ends_the_walk(self, db):
        """The last event decides, not the first — page one usually straddles."""
        bodies = {
            URL: _page(
                _card("/events/2026/8/7/a", "2026-08-07T20:00-0400"),
                _card("/events/2026/12/1/far", "2026-12-01T20:00-0500"),
                next_page="/venues/gulu-gulu-cafe?page=2",
            ),
            f"{URL}?page=2": _page(_card("/events/2027/1/1/further", "2027-01-01T20:00-0500")),
        }
        source, http = _make_source(db, bodies, horizon_days=45)

        results = source.fetch()

        assert http.requested == [URL]
        assert [c.id for c in results] == ["do617_gulu_gulu:/events/2026/8/7/a"]

    def test_a_page_repeating_earlier_events_ends_the_walk(self, db):
        """Sites commonly serve page 1 for any out-of-range page number."""
        repeated = _card("/events/2026/8/7/a", "2026-08-07T20:00-0400")
        bodies = {
            URL: _page(repeated, next_page="/venues/gulu-gulu-cafe?page=2"),
            f"{URL}?page=2": _page(repeated, next_page="/venues/gulu-gulu-cafe?page=3"),
        }
        source, http = _make_source(db, bodies)

        results = source.fetch()

        assert http.requested == [URL, f"{URL}?page=2"]
        assert [c.id for c in results] == ["do617_gulu_gulu:/events/2026/8/7/a"]

    def test_the_walk_never_exceeds_max_pages(self, db):
        """A pagination bug on somebody else's server is not our crawl."""
        bodies = {}
        for page in range(1, 12):
            url = URL if page == 1 else f"{URL}?page={page}"
            bodies[url] = _page(
                _card(f"/events/2026/8/7/e{page}", "2026-08-07T20:00-0400"),
                next_page=f"/venues/gulu-gulu-cafe?page={page + 1}",
            )
        source, http = _make_source(db, bodies, max_pages=3)

        source.fetch()

        assert len(http.requested) == 3

    def test_an_empty_venue_page_yields_nothing_without_error(self, db):
        source, http = _make_source(db, {URL: _page()})

        assert source.fetch() == []
        assert http.requested == [URL]
