"""Unit tests for VeeziSessionsSource."""

from __future__ import annotations

import io
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest

from src.storage.memory.http_cache import InMemoryHttpCache
from src.config import FeedConfig
from src.ingestion.candidate_id import derive_content_id
from src.ingestion.cinemas.veezi_source import VeeziSessionsSource
from src.models.event_candidate import EventCandidate
from src.storage.sqlite.connection import init_db
from src.utils.logging import get_logger
from tests.support.network import fetcher_for

FIXED_NOW = datetime(2026, 8, 7, 6, 0, tzinfo=timezone.utc)  # 02:00 in New York
TOKEN = "mz33me119qs1fn6sympyf41n2w"
URL = f"https://ticketing.useast.veezi.com/sessions/?siteToken={TOKEN}"


@pytest.fixture
def cache():
    """No database: these tests are about conditional requests."""
    return InMemoryHttpCache()


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


def _make_source(cache, body=_ONE_SHOWING, now=FIXED_NOW, session=None,
                 content_ids=False, **overrides):
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
        fetcher=fetcher_for(
            session if session is not None else _session(body),
            urls=URL,
            http_cache=cache,
            now=now,
        ),
        get_now=lambda: now,
        logger=get_logger("test", stream=io.StringIO()),
        timezone_name="America/New_York",
        uses_content_id=lambda source: content_ids,
    )


class TestMapping:
    def test_returns_event_candidates(self, cache):
        results = _make_source(cache).fetch()

        assert len(results) == 1
        assert isinstance(results[0], EventCandidate)

    def test_the_id_derives_from_the_session_and_is_stable(self, cache):
        """A nightly refetch must update its rows, not duplicate them."""
        first = _make_source(cache).fetch()[0]
        later = _make_source(cache, now=FIXED_NOW + timedelta(days=1)).fetch()[0]

        assert first.id == later.id == "cinemasalem:38750"

    def test_distinct_showings_get_distinct_ids(self, cache):
        body = _page(
            ("The Odyssey", "Friday 7, August", "38750", "3:30 PM"),
            ("The Odyssey", "Friday 7, August", "38754", "7:00 PM"),
        )

        results = _make_source(cache, body=body).fetch()

        assert len({c.id for c in results}) == 2

    def test_the_venue_and_city_come_from_config(self, cache):
        """The page names the film; only config knows which cinema it is."""
        candidate = _make_source(cache).fetch()[0]

        assert candidate.venue == "CinemaSalem"
        assert candidate.location == "Salem"

    def test_the_booking_link_is_carried(self, cache):
        candidate = _make_source(cache).fetch()[0]

        assert candidate.url.endswith(f"purchase/38750?siteToken={TOKEN}")

    def test_the_start_is_the_instant_of_7pm_in_the_cinemas_zone(self, cache):
        """A naive wall clock would shift by hours the moment it was localised.

        The offset it is *stored* under is a different question — the candidate
        canonicalises to UTC, so equality is asserted against the instant the
        cinema meant rather than against a representation.
        """
        candidate = _make_source(cache).fetch()[0]

        assert candidate.start_time.tzinfo is not None
        assert candidate.start_time == datetime(
            2026, 8, 7, 19, 0, tzinfo=timezone(timedelta(hours=-4))
        )

    def test_source_and_source_type_come_from_config(self, cache):
        candidate = _make_source(cache, source_type="movies_veezi").fetch()[0]

        assert candidate.source == "cinemasalem"
        assert candidate.source_type == "movies_veezi"

    def test_no_candidate_claims_a_published_date(self, cache):
        """A showtime listing carries no announcement date, and the lookback
        discards on that field."""
        assert _make_source(cache).fetch()[0].raw_published_at is None

    def test_the_title_is_the_film(self, cache):
        assert _make_source(cache).fetch()[0].title == "The Odyssey"


class TestPoliteness:
    def test_a_refetch_inside_the_interval_reuses_the_cache(self, cache):
        """Somebody else's server, and a nightly job that misbehaves gets blocked."""
        session = _session(_ONE_SHOWING)
        _make_source(cache, session=session).fetch()
        calls_after_first = session.get.call_count

        _make_source(
            cache, session=session, now=FIXED_NOW + timedelta(hours=1)
        ).fetch()

        assert calls_after_first == 1
        assert session.get.call_count == 1


class TestDegradation:
    def test_an_empty_page_yields_nothing(self, cache):
        assert _make_source(cache, body="<html><body></body></html>").fetch() == []

    def test_an_unreadable_row_does_not_lose_the_others(self, cache):
        body = _page(
            ("Broken", "Notaday 7, August", "1", "7:00 PM"),
            ("Fine", "Friday 7, August", "2", "8:00 PM"),
        )

        results = _make_source(cache, body=body).fetch()

        assert [c.title for c in results] == ["Fine"]


class TestWhenTheSessionIdsAreNotTrusted:
    """A Veezi `session_id` is the better key while it holds — it survives a
    retitling, and it tells one screen from another. This is the path for a
    cinema whose session ids stop identifying anything, which nothing but
    measurement can reveal.
    """

    def test_a_re_minted_session_id_yields_the_same_id(self, cache):
        first = _make_source(cache, content_ids=True).fetch()[0]
        renumbered = _page(("The Odyssey", "Friday 7, August", "99999", "7:00 PM"))

        later = _make_source(
            cache, body=renumbered, content_ids=True,
            now=FIXED_NOW + timedelta(days=1),
        ).fetch()[0]

        assert first.id == later.id

    def test_the_session_id_is_not_in_the_id_at_all(self, cache):
        assert "38750" not in _make_source(cache, content_ids=True).fetch()[0].id

    def test_two_showings_of_one_film_stay_distinct(self, cache):
        """Same film, same cinema, different times. Without the start they
        collapse into one candidate and the later showing disappears."""
        body = _page(
            ("The Odyssey", "Friday 7, August", "38750", "3:30 PM"),
            ("The Odyssey", "Friday 7, August", "38754", "7:00 PM"),
        )

        results = _make_source(cache, body=body, content_ids=True).fetch()

        assert len({c.id for c in results}) == 2

    def test_every_candidate_is_keyed_on_its_own_stored_fields(self, cache):
        """The id must be a function of what the row holds, not of anything on
        the way to it — the same invariant the re-key verifies."""
        for candidate in _make_source(cache, content_ids=True).fetch():
            assert candidate.id == derive_content_id(
                source=candidate.source,
                title=candidate.title,
                venue=candidate.venue,
                start=candidate.start_time,
            )

    def test_the_id_is_stable_across_a_refetch(self, cache):
        first = _make_source(cache, content_ids=True).fetch()[0]
        later = _make_source(
            cache, content_ids=True, now=FIXED_NOW + timedelta(days=1)
        ).fetch()[0]

        assert first.id == later.id
