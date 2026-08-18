"""PEM ingestion against the real captured events page.

No network: the fixture is `pem.org/events` as returned on 2026-08-07, reduced
to the two JSON-LD blocks it publishes plus a little chrome.

PEM was written off twice before this — feed autodiscovery finds nothing on the
site, and its Do617 venue page holds a deep archive of past events. Neither
probe could see the structured data sitting in the page itself.
"""

import io
from datetime import datetime, time, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.config import FeedConfig
from src.ingestion.aggregators.jsonld_listing import parse_jsonld_events
from src.ingestion.aggregators.jsonld_source import JsonLdEventSource
from src.models.timing import EXACT
from src.storage.sqlite.connection import init_db
from src.utils.logging import get_logger
from tests.support.network import fetcher_for

FIXTURE = Path(__file__).parent.parent / "fixtures" / "pem_events.html"
URL = "https://www.pem.org/events"
EASTERN = timezone(timedelta(hours=-4))

#: The day the page was captured; its listings run forward from it.
FIXED_NOW = datetime(2026, 8, 8, 10, 0, tzinfo=timezone.utc)


@pytest.fixture
def db(tmp_path):
    path = tmp_path / "test.db"
    init_db(path)
    return path


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


def _source(db, horizon_days=45):
    http = _FakeSession(FIXTURE.read_text())
    source = JsonLdEventSource(
        config=FeedConfig(name="pem", url=URL, source_type="pem"),
        fetcher=fetcher_for(
            http,
            urls=URL,
            now=FIXED_NOW,
        ),
        get_now=lambda: FIXED_NOW,
        logger=get_logger("test", stream=io.StringIO()),
        timezone_name="America/New_York",
        horizon_days=horizon_days,
        day_starts_at=time(4, 0),
    )
    return source, http


class TestTheRealPage:
    def test_reads_every_event_the_page_publishes(self):
        events = parse_jsonld_events(FIXTURE.read_text())

        assert len(events) == 97

    def test_events_run_from_the_capture_date_forward(self):
        events = parse_jsonld_events(FIXTURE.read_text())

        starts = sorted(event.start for event in events)
        assert starts[0] == datetime(2026, 8, 8, 11, 0, tzinfo=EASTERN)
        assert starts[-1].year == 2026

    def test_every_event_names_a_venue_and_address(self):
        events = parse_jsonld_events(FIXTURE.read_text())

        located = [e for e in events if e.venue and e.address]
        assert len(located) == len(events)


class TestIngestion:
    def test_one_request_yields_the_events_in_the_window(self, db):
        source, http = _source(db)

        candidates = source.fetch()

        assert http.requested == [URL]
        assert 0 < len(candidates) <= 97

    def test_candidates_carry_what_the_markup_stated(self, db):
        source, _ = _source(db)

        first = source.fetch()[0]
        assert first.title == "Drop-in Art Making: Chinese Painting Traditions"
        assert first.start_time == datetime(2026, 8, 8, 11, 0, tzinfo=EASTERN)
        assert first.venue == "Peabody Essex Museum"
        assert first.timing == EXACT
        assert first.url.startswith("https://www.pem.org/events/")

    def test_events_past_the_horizon_are_left_behind(self, db):
        """The page runs to late November, well beyond a 45-night window."""
        source, _ = _source(db, horizon_days=45)

        candidates = source.fetch()
        assert len(candidates) < 97
        assert all(
            c.start_time < datetime(2026, 9, 22, 4, 0, tzinfo=EASTERN) for c in candidates
        )

    def test_every_candidate_has_a_distinct_id(self, db):
        source, _ = _source(db)

        ids = [c.id for c in source.fetch()]
        assert len(set(ids)) == len(ids)
