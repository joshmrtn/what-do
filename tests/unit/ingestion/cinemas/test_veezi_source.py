"""Unit tests for VeeziSessionsSource."""

from __future__ import annotations

import io
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest

from src.config import FeedConfig
from src.ingestion.cinemas.veezi_source import VeeziSessionsSource
from src.models.event_candidate import EventCandidate
from src.storage.db import init_db
from src.utils.logging import get_logger

FIXED_NOW = datetime(2026, 8, 7, 6, 0, tzinfo=timezone.utc)  # 02:00 in New York
TOKEN = "mz33me119qs1fn6sympyf41n2w"
URL = f"https://ticketing.useast.veezi.com/sessions/?siteToken={TOKEN}"


@pytest.fixture
def db(tmp_path):
    path = tmp_path / "test.db"
    init_db(path)
    return path


def _page(*rows: tuple[str, str, str, str]) -> str:
    """Build a sessions page from (title, day heading, session id, time) rows."""
    films = ""
    for title, heading, session_id, when in rows:
        films += f"""
        <div class="film ">
          <div><h3 class="title">{title}</h3>
            <div class="sessions">
              <div class="date-container"><h4 class="date">{heading}</h4>
                <ul class="session-times"><li>
                  <a href="https://ticketing.useast.veezi.com/purchase/{session_id}?siteToken={TOKEN}">
                    <time>{when}</time></a></li></ul>
              </div>
            </div>
          </div>
        </div>"""
    return f"<html><body>{films}</body></html>"


_ONE_SHOWING = _page(("The Odyssey", "Friday 7, August", "38750", "7:00 PM"))


def _session(body=_ONE_SHOWING):
    response = MagicMock()
    response.status_code = 200
    response.text = body
    response.headers = {}
    response.raise_for_status.return_value = None
    http = MagicMock()
    http.get.return_value = response
    return http


def _make_source(db, body=_ONE_SHOWING, now=FIXED_NOW, **overrides):
    settings = {
        "name": "cinemasalem",
        "url": URL,
        "source_type": "cinema_veezi",
        "min_fetch_interval_hours": 6.0,
        "venue": "CinemaSalem",
        "city": "Salem",
    }
    settings.update(overrides)

    return VeeziSessionsSource(
        config=FeedConfig(**settings),
        db_path=db,
        session=_session(body),
        get_now=lambda: now,
        logger=get_logger("test", stream=io.StringIO()),
        timezone_name="America/New_York",
    )


class TestMapping:
    def test_returns_event_candidates(self, db):
        results = _make_source(db).fetch()

        assert len(results) == 1
        assert isinstance(results[0], EventCandidate)

    def test_the_id_derives_from_the_session_and_is_stable(self, db):
        """A nightly refetch must update its rows, not duplicate them."""
        first = _make_source(db).fetch()[0]
        later = _make_source(db, now=FIXED_NOW + timedelta(days=1)).fetch()[0]

        assert first.id == later.id == "cinemasalem:38750"

    def test_distinct_showings_get_distinct_ids(self, db):
        body = _page(
            ("The Odyssey", "Friday 7, August", "38750", "3:30 PM"),
            ("The Odyssey", "Friday 7, August", "38754", "7:00 PM"),
        )

        results = _make_source(db, body=body).fetch()

        assert len({c.id for c in results}) == 2

    def test_the_venue_and_city_come_from_config(self, db):
        """The page names the film; only config knows which cinema it is."""
        candidate = _make_source(db).fetch()[0]

        assert candidate.venue == "CinemaSalem"
        assert candidate.location == "Salem"

    def test_the_booking_link_is_carried(self, db):
        candidate = _make_source(db).fetch()[0]

        assert candidate.url.endswith(f"purchase/38750?siteToken={TOKEN}")

    def test_the_start_is_aware_in_the_cinemas_zone(self, db):
        """A naive wall clock would shift by hours the moment it was localised."""
        candidate = _make_source(db).fetch()[0]

        assert candidate.start_time.tzinfo is not None
        assert candidate.start_time.hour == 19
        assert candidate.start_time.utcoffset() == timedelta(hours=-4)

    def test_source_and_source_type_come_from_config(self, db):
        candidate = _make_source(db, source_type="movies_veezi").fetch()[0]

        assert candidate.source == "cinemasalem"
        assert candidate.source_type == "movies_veezi"

    def test_no_candidate_claims_a_published_date(self, db):
        """A showtime listing carries no announcement date, and the lookback
        discards on that field."""
        assert _make_source(db).fetch()[0].raw_published_at is None

    def test_the_title_is_the_film(self, db):
        assert _make_source(db).fetch()[0].title == "The Odyssey"


class TestPoliteness:
    def test_a_refetch_inside_the_interval_reuses_the_cache(self, db):
        """Somebody else's server, and a nightly job that misbehaves gets blocked."""
        source = _make_source(db)
        source.fetch()
        calls_after_first = source._session.get.call_count

        again = _make_source(db, now=FIXED_NOW + timedelta(hours=1))
        again.fetch()

        assert calls_after_first == 1
        assert again._session.get.call_count == 0


class TestDegradation:
    def test_an_empty_page_yields_nothing(self, db):
        assert _make_source(db, body="<html><body></body></html>").fetch() == []

    def test_an_unreadable_row_does_not_lose_the_others(self, db):
        body = _page(
            ("Broken", "Notaday 7, August", "1", "7:00 PM"),
            ("Fine", "Friday 7, August", "2", "8:00 PM"),
        )

        results = _make_source(db, body=body).fetch()

        assert [c.title for c in results] == ["Fine"]
