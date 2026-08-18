"""End-to-end checks for the HTML listing source against a real captured page.

Uses a trimmed capture of northshorenightout.com rather than the network.
Assertions are on shape and invariants — the fixture ages, and pinning one band's
name would fail for no useful reason.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import io
import sqlite3
import zoneinfo

import pytest

from src.storage.memory.http_cache import InMemoryHttpCache
from src.config import (
    AppConfig,
    FeedConfig,
    LocationConfig,
    ScrapingConfig,
    VenueDiscoveryConfig,
)
from src.ingestion.calendars.html_source import HtmlListingSource
from src.normalization.service import NormalizationService
from src.storage.sqlite.connection import init_db
from src.utils.logging import get_logger
from tests.support.network import fetcher_for

FIXTURE = Path("tests/fixtures/northshorenightout.html")
EASTERN = zoneinfo.ZoneInfo("America/New_York")
#: The page was captured on 2026-08-05, so its headings resolve against that day.
FIXED_NOW = datetime(2026, 8, 5, 12, 0, tzinfo=timezone.utc)
URL = "https://listings.example.com/"


@pytest.fixture
def db(tmp_path):
    path = tmp_path / "test.db"
    init_db(path)
    return path


class _FakeResponse:
    status_code = 200
    headers: dict[str, str] = {}

    def __init__(self, body: str) -> None:
        self.text = body

    def raise_for_status(self) -> None:
        return None


class _FakeSession:
    def __init__(self, body: str) -> None:
        self._body = body
        self.calls = 0

    def get(self, url, headers=None, params=None, timeout=None):
        self.calls += 1
        return _FakeResponse(self._body)


@pytest.fixture
def candidates():
    source = HtmlListingSource(
        config=FeedConfig(
            name="northshorenightout",
            url=URL,
            source_type="northshorenightout",
        ),
        fetcher=fetcher_for(
            _FakeSession(FIXTURE.read_text(encoding="utf-8")),
            urls=URL,
            now=FIXED_NOW,
        ),
        tzname="America/New_York",
        get_now=lambda: FIXED_NOW,
        logger=get_logger("test", stream=io.StringIO()),
    )
    return source.fetch()


def test_the_whole_page_becomes_candidates(candidates):
    """199 event lines, one of which the page itself lists twice."""

    assert len(candidates) == 198


def test_every_candidate_is_identifiable(candidates):

    assert all(c.title for c in candidates)
    assert len({c.id for c in candidates}) == len(candidates)


def test_every_candidate_has_an_aware_start_time(candidates):

    assert all(c.start_time is not None for c in candidates)
    assert all(c.start_time.tzinfo is not None for c in candidates)


def test_the_listing_covers_far_more_venues_than_the_feed(candidates):
    """Breadth is the whole reason this source exists alongside the ICS one."""

    venues = {c.venue for c in candidates if c.venue}

    assert len(venues) > 50


def test_nearly_every_candidate_carries_a_venue(candidates):

    assert sum(1 for c in candidates if c.venue) / len(candidates) >= 0.95


def test_some_candidates_carry_event_links(candidates):
    """Links are what this source adds that the calendar feed cannot."""

    assert sum(1 for c in candidates if c.url) >= 20


def test_no_candidate_claims_a_published_date(candidates):

    assert all(c.raw_published_at is None for c in candidates)


def test_no_candidate_invents_a_description(candidates):
    """The listing publishes one line per event and no prose about any of them."""

    assert all(c.description is None for c in candidates)


def test_useful_categories_reach_the_metadata(candidates):
    carried = [c.metadata["listing_category"] for c in candidates
               if "listing_category" in c.metadata]

    assert len(carried) >= 100
    assert set(carried) <= {"Music", "Sports"}


def test_every_candidate_carries_an_authored_summary(candidates):
    assert all(c.summary for c in candidates)
    assert all(c.metadata["authored_summary"] is True for c in candidates)


def test_karaoke_and_trivia_lines_are_authored_not_extracted(candidates):
    authored = [c for c in candidates if c.metadata.get("authored_tags")]

    assert authored, "the fixture page carries karaoke and trivia lines"
    for candidate in authored:
        tags = {t.text for t in candidate.tags}
        assert ("karaoke" in tags) ^ ("trivia" in tags), candidate.title


def test_start_times_stay_within_the_days_the_page_shows(candidates):

    days = {c.start_time.astimezone(EASTERN).date().isoformat() for c in candidates}

    assert min(days) == "2026-08-05"
    assert max(days) == "2026-08-10"


def test_evening_events_keep_their_evening_hour(candidates):
    """A UTC mix-up would push a 7pm show into the small hours of the next day."""

    evening = [
        c for c in candidates
        if c.metadata.get("listing_category") == "Music"
    ]
    hours = {c.start_time.astimezone(EASTERN).hour for c in evening}

    assert hours
    assert all(0 <= h <= 23 for h in hours)
    assert any(17 <= h <= 22 for h in hours)


def test_a_second_fetch_reuses_the_cache(db):

    session = _FakeSession(FIXTURE.read_text(encoding="utf-8"))
    config = FeedConfig(
        name="northshorenightout", url=URL, source_type="northshorenightout"
    )

    # One cache across both fetches — the politeness floor is about state
    # surviving between them, which a per-call cache would erase.
    cache = InMemoryHttpCache()

    def _source():
        return HtmlListingSource(
            config=config,
            fetcher=fetcher_for(session, urls=URL, http_cache=cache, now=FIXED_NOW),
            tzname="America/New_York",
            get_now=lambda: FIXED_NOW,
            logger=get_logger("test", stream=io.StringIO()),
        )

    first = _source().fetch()
    second = _source().fetch()

    assert session.calls == 1
    assert [c.id for c in first] == [c.id for c in second]


def test_candidates_normalise_and_persist(db, candidates):
    """The real handoff: this source composes with the Phase 4 pipeline."""

    config = AppConfig(
        location=LocationConfig(
            latitude=42.52,
            longitude=-70.89,
            postal_code="01970",
            search_radius_miles=10,
            timezone="America/New_York",
        ),
        scraping=ScrapingConfig(),
        venue_discovery=VenueDiscoveryConfig(),
    )
    service = NormalizationService(
        config=config, logger=get_logger("test", stream=io.StringIO())
    )

    result = service.run(candidates, get_now=lambda: FIXED_NOW)

    assert result.normalized > 0

    events = [e for e in result.events if e.source_type == "northshorenightout"]

    assert len(events) == result.normalized
