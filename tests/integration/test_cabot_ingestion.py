"""Cabot ingestion against a real captured listing page.

No network: the fixture is a real response, served through a fake session. The
parser's rules were derived from this markup, so a page that changes shape
should fail here rather than silently ingest nothing at 2am.
"""

from __future__ import annotations

import io
from datetime import date, datetime, time, timezone
from pathlib import Path

import pytest

from src.config import FeedConfig
from src.ingestion.cinemas.cabot_listing import parse_cabot
from src.ingestion.cinemas.cabot_source import CabotListingSource
from src.storage.sqlite.connection import init_db
from src.utils.logging import get_logger
from tests.support.network import fetcher_for

FIXTURE = Path(__file__).parent.parent / "fixtures" / "cabot_whats_on.html"

#: The page was captured on this date, which its bare `7 Aug` resolves against.
FIXED_NOW = datetime(2026, 8, 7, 6, 0, tzinfo=timezone.utc)


class _FakeSession:
    def __init__(self, body: str) -> None:
        self._body = body
        self.requested: list[str] = []

    def get(self, url, headers=None, params=None, timeout=None):
        self.requested.append(url)
        # Only page one exists in this capture; anything further is empty, as a
        # well-behaved listing would answer.
        body = self._body if url.endswith("/whats-on/") else "<html><body></body></html>"

        class _Response:
            status_code = 200
            headers: dict[str, str] = {}
            text = body

            def raise_for_status(self) -> None:
                return None

        return _Response()


@pytest.fixture
def db(tmp_path):
    path = tmp_path / "test.db"
    init_db(path)
    return path


@pytest.fixture
def page() -> str:
    return FIXTURE.read_text(encoding="utf-8")


def test_the_page_yields_its_ten_events(page):
    events, total = parse_cabot(page, date(2026, 8, 7), want_total=True)

    assert len(events) == 10
    assert total == 88


def test_the_listing_is_ascending(page):
    events = parse_cabot(page, date(2026, 8, 7))

    assert [e.start.date() for e in events] == sorted(e.start.date() for e in events)


def test_a_run_of_dates_becomes_a_span(page):
    """`11 - 25 Aug` is a fortnight of drop-ins, not one night."""
    events = parse_cabot(page, date(2026, 8, 7))
    run = next(e for e in events if e.title == "Essex Improv Drop-In")

    assert run.start.date() == date(2026, 8, 11)
    assert run.end is not None and run.end.date() == date(2026, 8, 25)
    assert run.time_known is False


def test_the_listing_is_more_than_a_cinema(page):
    """Music, comedy and talks alongside the $1 Movie Series."""
    genres = {g for e in parse_cabot(page, date(2026, 8, 7)) for g in e.genres}

    assert {"Music", "Comedy", "Films", "Talk/Conversation"} <= genres


def test_every_event_is_identified_and_linked(page):
    events = parse_cabot(page, date(2026, 8, 7))

    assert all(e.event_id and e.url for e in events)
    assert len({e.event_id for e in events}) == len(events)


def test_the_adapter_maps_the_page_to_candidates(db, page):
    source = CabotListingSource(
        config=FeedConfig(
            name="cabot",
            url="https://thecabot.org/whats-on/",
            source_type="cabot",
            venue="The Cabot",
            city="Beverly",
        ),
        fetcher=fetcher_for(
            _FakeSession(page),
            urls="https://thecabot.org/whats-on/",
            now=FIXED_NOW,
        ),
        get_now=lambda: FIXED_NOW,
        logger=get_logger("test", stream=io.StringIO()),
        timezone_name="America/New_York",
        horizon_days=45,
        day_starts_at=time(4, 0),
        uses_content_id=lambda source: False,
    )

    candidates = source.fetch()

    assert len(candidates) == 10
    assert all(c.start_time.tzinfo is not None for c in candidates)
    assert all(c.venue for c in candidates)
    assert {c.venue for c in candidates} == {"The Cabot", "Off Cabot"}
