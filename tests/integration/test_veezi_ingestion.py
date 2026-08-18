"""Veezi ingestion against real captured pages.

No network: the fixtures are real responses from two cinemas' public ticketing
pages, served through a fake session. They exist because the parser's rules were
derived from this markup, and a page that changes shape should fail here rather
than silently ingest nothing at 2am.
"""

from __future__ import annotations

import io
from datetime import datetime, timezone
from pathlib import Path

import pytest

from src.config import FeedConfig
from src.ingestion.cinemas.veezi_source import VeeziSessionsSource
from src.storage.sqlite.connection import init_db
from src.utils.logging import get_logger
from tests.support.network import fetcher_for

FIXTURES = Path(__file__).parent.parent / "fixtures"

#: The pages were captured on this date, which the bare day headings resolve against.
FIXED_NOW = datetime(2026, 8, 7, 6, 0, tzinfo=timezone.utc)


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

    def get(self, url, headers=None, timeout=None):
        self.calls += 1
        return _FakeResponse(self._body)


@pytest.fixture
def db(tmp_path):
    path = tmp_path / "test.db"
    init_db(path)
    return path


def _candidates(db, fixture: str, venue: str, city: str):
    source = VeeziSessionsSource(
        config=FeedConfig(
            name=fixture,
            url=f"https://ticketing.useast.veezi.com/sessions/?siteToken={fixture}",
            source_type="cinema_veezi",
            venue=venue,
            city=city,
        ),
        fetcher=fetcher_for(
            _FakeSession((FIXTURES / f"veezi_{fixture}.html").read_text(encoding="utf-8")),
            urls=f"https://ticketing.useast.veezi.com/sessions/?siteToken={fixture}",
            now=FIXED_NOW,
        ),
        get_now=lambda: FIXED_NOW,
        logger=get_logger("test", stream=io.StringIO()),
        timezone_name="America/New_York",
    )
    return source.fetch()


@pytest.fixture
def cinemasalem(db):
    return _candidates(db, "cinemasalem", "CinemaSalem", "Salem")


@pytest.fixture
def warwick(db):
    return _candidates(db, "warwick", "Warwick Cinema", "Marblehead")


def test_cinemasalem_yields_its_showings(cinemasalem):
    """35 films across 85 showings, from a page that lists each one twice over."""

    assert len(cinemasalem) == 85
    assert len({c.title for c in cinemasalem}) == 35


def test_warwick_yields_its_showings(warwick):
    assert len(warwick) == 72
    assert len({c.title for c in warwick}) == 3


def test_every_showing_is_uniquely_identified(cinemasalem, warwick):
    """The session id is the cinema's own key, and dedup rests on it."""

    for candidates in (cinemasalem, warwick):
        assert len({c.id for c in candidates}) == len(candidates)


def test_every_showing_has_an_aware_start(cinemasalem):
    """A naive wall clock would shift by hours once localised."""

    assert all(c.start_time is not None for c in cinemasalem)
    assert all(c.start_time.tzinfo is not None for c in cinemasalem)


def test_every_showing_carries_a_booking_link(cinemasalem):
    assert all(c.url and "/purchase/" in c.url for c in cinemasalem)


def test_every_showing_is_attributed_to_its_cinema(cinemasalem, warwick):
    """The page names only the film, so venue comes from config."""

    assert all(c.venue == "CinemaSalem" for c in cinemasalem)
    assert all(c.venue == "Warwick Cinema" for c in warwick)


def test_no_showing_claims_a_published_date(cinemasalem):
    assert all(c.raw_published_at is None for c in cinemasalem)


def test_the_two_cinemas_do_not_collide(cinemasalem, warwick):
    """Ids are namespaced by source, so two cinemas can share a session number."""

    assert not ({c.id for c in cinemasalem} & {c.id for c in warwick})
