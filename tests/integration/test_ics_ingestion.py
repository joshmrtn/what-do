"""End-to-end checks for the ICS source against a real captured feed.

Uses the saved North Shore Night Out calendar rather than the network. Assertions
are on shape and invariants, never on individual events — the fixture ages, and a
test that pins one band's name would fail for no useful reason.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import io
import sqlite3

import pytest

from src.storage.memory.http_cache import InMemoryHttpCache
from src.config import (
    AppConfig,
    FeedConfig,
    LocationConfig,
    ScrapingConfig,
    VenueDiscoveryConfig,
)
from src.ingestion.calendars.ics_source import IcsCalendarSource
from src.normalization.service import NormalizationService
from src.storage.sqlite.connection import init_db
from src.utils.logging import get_logger
from tests.support.network import fetcher_for

FIXTURE = Path("tests/fixtures/northshorenightout.ics")
FIXED_NOW = datetime(2026, 8, 5, 2, 0, tzinfo=timezone.utc)
URL = "https://calendar.example.com/public/basic.ics"


@pytest.fixture
def db(tmp_path):
    path = tmp_path / "test.db"
    init_db(path)
    return path


class _FakeResponse:
    """Serves the captured feed in place of a live request."""

    status_code = 200
    headers: dict[str, str] = {"ETag": 'W/"fixture"'}

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
    source = IcsCalendarSource(
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
        get_now=lambda: FIXED_NOW,
        logger=get_logger("test", stream=io.StringIO()),
    )
    return source.fetch()


def test_the_feed_becomes_candidates_inside_the_horizon(candidates):
    """The feed runs ~39 days out; the default horizon is 30.

    Of 144 events, 125 fall inside the default 45-night window; the rest have
    already happened. The window is aligned to the night rather than the run
    instant, so `horizon_days` counts whole nights from the start of this one.
    """

    assert len(candidates) == 125


def test_every_candidate_is_identifiable(candidates):

    assert all(c.title for c in candidates)
    assert len({c.id for c in candidates}) == len(candidates)


def test_every_candidate_has_an_aware_start_time(candidates):
    """A naive timestamp would silently shift by hours once localised."""

    assert all(c.start_time is not None for c in candidates)
    assert all(c.start_time.tzinfo is not None for c in candidates)


def test_most_candidates_carry_a_venue(candidates):
    """The [Venue, City] convention is the feed's, so this tracks its health."""

    with_venue = [c for c in candidates if c.venue]

    assert len(with_venue) / len(candidates) >= 0.9


def test_end_times_are_present_where_the_feed_declares_them(candidates):

    assert sum(1 for c in candidates if c.end_time) == 107


def test_no_candidate_claims_a_published_date(candidates):

    assert all(c.raw_published_at is None for c in candidates)


def test_descriptions_carry_no_residual_markup(candidates):

    described = [c.description for c in candidates if c.description]

    assert described
    assert not any("<br" in d or "</p>" in d or "&nbsp;" in d for d in described)


def test_a_second_fetch_reuses_the_cache_without_touching_the_network(db):
    """The politeness floor, proven end to end rather than by unit stub."""

    session = _FakeSession(FIXTURE.read_text(encoding="utf-8"))
    config = FeedConfig(
        name="northshorenightout", url=URL, source_type="northshorenightout"
    )

    # One cache across both fetches — the politeness floor is about state
    # surviving between them, which a per-call cache would erase.
    cache = InMemoryHttpCache()

    def _source():
        return IcsCalendarSource(
            config=config,
            fetcher=fetcher_for(
                session,
                urls=URL,
                http_cache=cache,
                now=FIXED_NOW,
            ),
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
    assert all(e.title for e in events)
    assert all(e.start_time for e in events)
