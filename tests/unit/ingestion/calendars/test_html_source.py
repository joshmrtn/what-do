"""Unit tests for HtmlListingSource."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock
import io
import zoneinfo

import pytest

from src.config import FeedConfig
from src.ingestion.calendars.html_source import HtmlListingSource
from src.models.event_candidate import EventCandidate
from src.storage.db import init_db
from src.storage.sqlite.http_cache import SqliteHttpCache
from src.utils.logging import get_logger

EASTERN = zoneinfo.ZoneInfo("America/New_York")
FIXED_NOW = datetime(2026, 8, 5, 12, 0, tzinfo=timezone.utc)
URL = "https://listings.example.com/"


@pytest.fixture
def db(tmp_path):
    path = tmp_path / "test.db"
    init_db(path)
    return path


_PAGE = (
    "<body><div class='sqs-html-content'>"
    '<p class="sqsrte-large"><strong>Wednesday, August 5</strong></p>'
    "<p><strong>Music</strong></p>"
    '<p class="sqsrte-small">6:30 PM - Jazz Night - Joy Nest - Newburyport</p>'
    '<p class="sqsrte-small">7:00 PM -<a href="https://venue.test/e/1"><u>Alex Anthony</u>'
    "</a> - Minglewood Harborside - Gloucester</p>"
    "</div></body>"
)


def _response(body=_PAGE, status=200, headers=None):
    response = MagicMock()
    response.status_code = status
    response.text = body
    response.headers = headers or {}
    response.raise_for_status.return_value = None
    return response


def _session(response=None):
    session = MagicMock()
    session.get.return_value = response if response is not None else _response()
    return session


def _make_source(db, session=None, now=FIXED_NOW, stream=None, **overrides):
    settings = {
        "name": "northshorenightout",
        "url": URL,
        "source_type": "northshorenightout",
        "min_fetch_interval_hours": 6.0,
    }
    settings.update(overrides)

    return HtmlListingSource(
        config=FeedConfig(**settings),
        db_path=db,
        tzname="America/New_York",
        session=session or _session(),
        get_now=lambda: now,
        logger=get_logger("test", stream=stream or io.StringIO()),
    )


def test_returns_event_candidates(db):

    results = _make_source(db).fetch()

    assert len(results) == 2
    assert all(isinstance(c, EventCandidate) for c in results)


def test_maps_the_listing_fields(db):

    first = _make_source(db).fetch()[0]

    assert first.title == "Jazz Night"
    assert first.venue == "Joy Nest"
    assert first.location == "Newburyport"
    assert first.source == "northshorenightout"
    assert first.source_type == "northshorenightout"


def test_start_time_is_localised_not_naive(db):
    """A naive time would be compared against an aware clock during ingestion."""

    first = _make_source(db).fetch()[0]

    assert first.start_time.tzinfo is not None
    assert first.start_time == datetime(2026, 8, 5, 18, 30, tzinfo=EASTERN)


def test_local_wall_clock_time_is_preserved(db):
    """6:30 PM on the page must stay 6:30 PM locally, not shift by the UTC offset."""

    first = _make_source(db).fetch()[0]

    assert first.start_time.astimezone(EASTERN).hour == 18
    assert first.start_time.astimezone(EASTERN).minute == 30


def test_event_link_becomes_the_candidate_url(db):

    linked = _make_source(db).fetch()[1]

    assert linked.title == "Alex Anthony"
    assert linked.url == "https://venue.test/e/1"


def test_unlinked_event_has_no_url(db):

    assert _make_source(db).fetch()[0].url is None


def test_a_useful_category_is_carried_as_data_not_as_a_description(db):
    """A section heading is not something the source published about the event.

    Storing it as a description made LLM Pass 1 read a taxonomy label as event
    copy — `Trivia` under `Karaoke & trivia` came back tagged `karaoke 1.0`.
    """
    first = _make_source(db).fetch()[0]

    assert first.description is None
    assert first.metadata["listing_category"] == "Music"


def test_a_category_naming_two_activities_is_dropped(db):
    """Every event under `Karaoke & trivia` already says which one it is.

    The heading adds nothing the title does not, and costs a wrong tag on all
    of them.
    """
    page = (
        "<body><div>"
        '<p class="sqsrte-large"><strong>Wednesday, August 5</strong></p>'
        "<p><strong>Karaoke &amp; trivia</strong></p>"
        '<p class="sqsrte-small">7:00 PM - Trivia - The James - Essex</p>'
        "</div></body>"
    )

    candidate = _make_source(db, session=_session(_response(page))).fetch()[0]

    assert candidate.description is None
    assert "listing_category" not in candidate.metadata


def test_the_other_category_is_dropped_as_noise(db):
    page = (
        "<body><div>"
        '<p class="sqsrte-large"><strong>Wednesday, August 5</strong></p>'
        "<p><strong>Other</strong></p>"
        '<p class="sqsrte-small">7:00 PM - Dart Night - Notch - Salem</p>'
        "</div></body>"
    )

    candidate = _make_source(db, session=_session(_response(page))).fetch()[0]

    assert "listing_category" not in candidate.metadata


def test_sports_is_carried_like_music(db):
    page = (
        "<body><div>"
        '<p class="sqsrte-large"><strong>Wednesday, August 5</strong></p>'
        "<p><strong>Sports</strong></p>"
        '<p class="sqsrte-small">7:00 PM - Sox Game - The Babe - Salem</p>'
        "</div></body>"
    )

    candidate = _make_source(db, session=_session(_response(page))).fetch()[0]

    assert candidate.metadata["listing_category"] == "Sports"


def test_no_category_leaves_the_description_empty(db):
    page = (
        "<body><div>"
        '<p class="sqsrte-large"><strong>Wednesday, August 5</strong></p>'
        '<p class="sqsrte-small">7:00 PM - Show - V - C</p>'
        "</div></body>"
    )

    candidate = _make_source(db, session=_session(_response(page))).fetch()[0]

    assert candidate.description is None
    assert "listing_category" not in candidate.metadata


def test_ids_are_stable_across_fetches(db):
    """The listing has no UID, so identity must derive from stable content."""

    first = _make_source(db).fetch()
    later = _make_source(db, now=FIXED_NOW + timedelta(days=1)).fetch()

    assert [c.id for c in first] == [c.id for c in later]


def test_ids_are_unique_within_a_page(db):

    candidates = _make_source(db).fetch()

    assert len({c.id for c in candidates}) == len(candidates)


def test_a_repeated_listing_line_yields_one_candidate(db):
    """Listing pages do repeat events; two objects sharing an id would mislead."""

    repeated = (
        "<body><div>"
        '<p class="sqsrte-large"><strong>Wednesday, August 5</strong></p>'
        '<p class="sqsrte-small">8:00 PM - Forever Beatles - Chianti - Beverly</p>'
        '<p class="sqsrte-small">8:00 PM - Forever Beatles - Chianti - Beverly</p>'
        "</div></body>"
    )

    candidates = _make_source(db, session=_session(_response(repeated))).fetch()

    assert len(candidates) == 1


def test_id_changes_when_the_event_does(db):
    """The fetch interval is disabled here so the cache cannot mask the change."""
    page = _PAGE.replace("Jazz Night", "Blues Night")

    original = _make_source(db, min_fetch_interval_hours=0.0).fetch()[0]
    changed = _make_source(
        db, session=_session(_response(page)), min_fetch_interval_hours=0.0
    ).fetch()[0]

    assert original.id != changed.id


def test_raw_published_at_is_never_set(db):

    assert all(c.raw_published_at is None for c in _make_source(db).fetch())


def test_end_time_is_absent_because_the_listing_has_none(db):

    assert all(c.end_time is None for c in _make_source(db).fetch())


def test_discovered_at_uses_the_injected_clock(db):

    assert _make_source(db).fetch()[0].discovered_at == FIXED_NOW


def test_the_listing_date_comes_from_local_time_not_utc(db):
    """At 00:30 UTC it is still the previous evening locally, and the page shows that."""

    page = (
        "<body><div>"
        '<p class="sqsrte-large"><strong>Tuesday, August 4</strong></p>'
        '<p class="sqsrte-small">9:00 PM - Late Show - V - C</p>'
        "</div></body>"
    )
    just_after_utc_midnight = datetime(2026, 8, 5, 0, 30, tzinfo=timezone.utc)

    candidates = _make_source(
        db, session=_session(_response(page)), now=just_after_utc_midnight
    ).fetch()

    assert len(candidates) == 1
    assert candidates[0].start_time.astimezone(EASTERN).date().isoformat() == "2026-08-04"


def test_within_the_fetch_interval_no_request_is_made(db):

    SqliteHttpCache(db).put(
        URL, body=_PAGE, etag=None, last_modified=None,
        fetched_at=FIXED_NOW - timedelta(hours=1),
    )
    session = _session()

    candidates = _make_source(db, session=session).fetch()

    session.get.assert_not_called()
    assert len(candidates) == 2


def test_request_identifies_itself_and_sets_a_timeout(db):

    session = _session()
    _make_source(db, session=session).fetch()

    kwargs = session.get.call_args.kwargs
    assert "what-do" in kwargs["headers"]["User-Agent"].lower()
    assert kwargs["timeout"] > 0


def test_http_error_propagates_without_retrying(db):

    response = _response()
    response.raise_for_status.side_effect = RuntimeError("503")
    session = _session(response)

    with pytest.raises(RuntimeError):
        _make_source(db, session=session).fetch()

    session.get.assert_called_once()


class TestAuthoredSummary:
    """The listing states one line; a summary of it is composition, not inference."""

    def test_the_summary_is_composed_from_the_parsed_fields(self, db):
        first = _make_source(db).fetch()[0]

        assert first.summary == "Jazz Night at Joy Nest in Newburyport"
        assert first.metadata["authored_summary"] is True

    def test_a_missing_city_drops_the_in_clause(self, db):
        page = (
            "<body><div>"
            '<p class="sqsrte-large"><strong>Wednesday, August 5</strong></p>'
            "<p><strong>Music</strong></p>"
            '<p class="sqsrte-small">7:00 PM - Open Mic - The Pelican</p>'
            "</div></body>"
        )

        candidate = _make_source(db, session=_session(_response(page))).fetch()[0]

        assert candidate.summary == "Open Mic at The Pelican"


class TestAuthoredKaraokeAndTrivia:
    """`Karaoke` and `Trivia` need no model: the title is the whole activity."""

    def _fetch(self, db, title):
        page = (
            "<body><div>"
            '<p class="sqsrte-large"><strong>Wednesday, August 5</strong></p>'
            "<p><strong>Karaoke &amp; trivia</strong></p>"
            f'<p class="sqsrte-small">7:00 PM - {title} - The James - Essex</p>'
            "</div></body>"
        )
        return _make_source(db, session=_session(_response(page))).fetch()[0]

    def test_a_trivia_night_gets_authored_tags_and_no_karaoke(self, db):
        candidate = self._fetch(db, "Trivia")

        tags = {t.text for t in candidate.tags}
        assert "trivia" in tags
        assert not any("karaoke" in t for t in tags)
        assert candidate.metadata["authored_tags"] is True

    def test_a_karaoke_night_gets_authored_tags_and_no_trivia(self, db):
        candidate = self._fetch(db, "Karaoke")

        tags = {t.text for t in candidate.tags}
        assert "karaoke" in tags
        assert not any("trivia" in t for t in tags)

    def test_a_titled_variant_is_still_recognised(self, db):
        candidate = self._fetch(db, "Trivia w/ Lee Wolf")

        assert "trivia" in {t.text for t in candidate.tags}

    def test_an_unrecognised_line_under_the_heading_is_left_to_the_model(self, db):
        """`DJ Bingo Night` appears under the same heading and is neither."""
        candidate = self._fetch(db, "DJ Bingo Night")

        assert candidate.tags == []
        assert "authored_tags" not in candidate.metadata

    def test_a_music_event_is_never_given_authored_tags(self, db):
        first = _make_source(db).fetch()[0]

        assert first.tags == []
        assert "authored_tags" not in first.metadata
