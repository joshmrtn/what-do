"""Unit tests for IcsCalendarSource."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock
import io

import pytest

from src.config import FeedConfig
from src.ingestion.calendars.ics_source import IcsCalendarSource
from src.models.event_candidate import EventCandidate
from src.storage.db import init_db
from src.storage.http_cache import read_cache, write_cache
from src.utils.logging import get_logger

FIXED_NOW = datetime(2026, 8, 5, 2, 0, tzinfo=timezone.utc)
URL = "https://calendar.example.com/public/basic.ics"


@pytest.fixture
def db(tmp_path):
    path = tmp_path / "test.db"
    init_db(path)
    return path


def _calendar(*bodies: str) -> str:
    blocks = "".join(f"BEGIN:VEVENT\r\n{b}\r\nEND:VEVENT\r\n" for b in bodies)
    return f"BEGIN:VCALENDAR\r\nVERSION:2.0\r\n{blocks}END:VCALENDAR\r\n"


_ONE_EVENT = _calendar(
    "UID:abc123@google.com\r\n"
    "SUMMARY:​[The Rhumb Line\\, Gloucester\\, Music] Three Ply\r\n"
    "DTSTART:20260809T010000Z\r\n"
    "DTEND:20260809T040000Z\r\n"
    "STATUS:CONFIRMED"
)


def _response(body=_ONE_EVENT, status=200, headers=None):
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


def _make_source(db, session=None, now=FIXED_NOW, stream=None, **config_overrides):
    settings = {
        "name": "northshorenightout",
        "url": URL,
        "source_type": "northshorenightout",
        "min_fetch_interval_hours": 6.0,
    }
    settings.update(config_overrides)

    return IcsCalendarSource(
        config=FeedConfig(**settings),
        db_path=db,
        session=session or _session(),
        get_now=lambda: now,
        logger=get_logger("test", stream=stream or io.StringIO()),
    )


def test_returns_event_candidates(db):

    results = _make_source(db).fetch()

    assert len(results) == 1
    assert isinstance(results[0], EventCandidate)


def test_id_derives_from_uid_and_is_stable_across_fetches(db):
    """A nightly refetch must update its rows, not duplicate them."""

    first = _make_source(db).fetch()[0]
    later = _make_source(db, now=FIXED_NOW + timedelta(days=1)).fetch()[0]

    assert first.id == later.id
    assert "abc123@google.com" in first.id


def test_source_and_source_type_come_from_config(db):

    candidate = _make_source(db, source_type="community_calendar").fetch()[0]

    assert candidate.source == "northshorenightout"
    assert candidate.source_type == "community_calendar"


def test_venue_and_city_parse_out_of_the_title_prefix(db):

    candidate = _make_source(db).fetch()[0]

    assert candidate.title == "Three Ply"
    assert candidate.venue == "The Rhumb Line"
    assert candidate.location == "Gloucester"


def test_zero_width_space_is_stripped_from_the_title(db):

    candidate = _make_source(db).fetch()[0]

    assert "​" not in candidate.title
    assert "​" not in candidate.venue


def test_category_segment_is_prepended_to_the_description(db):
    """Category is real signal, but must not become a tag and bypass extraction."""

    candidate = _make_source(db).fetch()[0]

    assert candidate.description is not None
    assert candidate.description.startswith("Category: Music")


def test_two_segment_prefix_has_no_category(db):

    ics = _calendar(
        "UID:a@x\r\nSUMMARY:​[Bent Water Brewing CO.\\, Lynn] General Trivia\r\n"
        "DTSTART:20260806T230000Z"
    )
    candidate = _make_source(db, session=_session(_response(ics))).fetch()[0]

    assert candidate.title == "General Trivia"
    assert candidate.venue == "Bent Water Brewing CO."
    assert candidate.location == "Lynn"
    assert candidate.description is None


def test_description_keeps_its_own_text_below_the_category(db):

    ics = _calendar(
        "UID:a@x\r\nSUMMARY:​[Venue\\, City\\, Music] Show\r\n"
        "DESCRIPTION:Doors at 7.\r\nDTSTART:20260806T230000Z"
    )
    candidate = _make_source(db, session=_session(_response(ics))).fetch()[0]

    assert "Category: Music" in candidate.description
    assert "Doors at 7." in candidate.description


def test_unconventional_summary_keeps_the_event_and_warns(db):
    """A convention change must cost venue attribution, never the event."""

    stream = io.StringIO()
    ics = _calendar("UID:a@x\r\nSUMMARY:Just A Title\r\nDTSTART:20260806T230000Z")

    candidates = _make_source(
        db, session=_session(_response(ics)), stream=stream
    ).fetch()

    assert len(candidates) == 1
    assert candidates[0].title == "Just A Title"
    assert candidates[0].venue is None
    assert "Just A Title" in stream.getvalue()


def test_timestamps_map_to_start_and_end(db):

    candidate = _make_source(db).fetch()[0]

    assert candidate.start_time == datetime(2026, 8, 9, 1, 0, tzinfo=timezone.utc)
    assert candidate.end_time == datetime(2026, 8, 9, 4, 0, tzinfo=timezone.utc)


def test_raw_published_at_is_never_set(db):
    """CREATED tracks calendar rebuilds, and the field drives the lookback discard."""

    ics = _calendar(
        "UID:a@x\r\nSUMMARY:​[V\\, C] Show\r\n"
        "CREATED:20200101T000000Z\r\nDTSTART:20260806T230000Z"
    )
    candidate = _make_source(db, session=_session(_response(ics))).fetch()[0]

    assert candidate.raw_published_at is None


def test_cancelled_events_are_skipped(db):

    ics = _calendar(
        "UID:a@x\r\nSUMMARY:​[V\\, C] Gone\r\n"
        "DTSTART:20260806T230000Z\r\nSTATUS:CANCELLED",
        "UID:b@x\r\nSUMMARY:​[V\\, C] Still On\r\nDTSTART:20260806T230000Z",
    )
    candidates = _make_source(db, session=_session(_response(ics))).fetch()

    assert [c.title for c in candidates] == ["Still On"]


def test_html_in_the_description_is_converted_not_dropped(db):

    ics = _calendar(
        "UID:a@x\r\nSUMMARY:​[V\\, C] Show\r\n"
        "DESCRIPTION:Boston!<br>You will<br>see <a href=\"https://x.test/t\">tickets</a>\r\n"
        "DTSTART:20260806T230000Z"
    )
    candidate = _make_source(db, session=_session(_response(ics))).fetch()[0]

    assert "Boston!\nYou will" in candidate.description
    assert "https://x.test/t" in candidate.description


def test_url_property_becomes_the_candidate_url(db):

    ics = _calendar(
        "UID:a@x\r\nSUMMARY:​[V\\, C] Show\r\n"
        "URL:https://example.com/event\r\nDTSTART:20260806T230000Z"
    )
    candidate = _make_source(db, session=_session(_response(ics))).fetch()[0]

    assert candidate.url == "https://example.com/event"


def test_event_without_a_uid_is_skipped(db):
    """Without a stable key the candidate would duplicate on every run."""

    ics = _calendar("SUMMARY:​[V\\, C] Anonymous\r\nDTSTART:20260806T230000Z")

    assert _make_source(db, session=_session(_response(ics))).fetch() == []


def test_discovered_at_uses_the_injected_clock(db):

    candidate = _make_source(db).fetch()[0]

    assert candidate.discovered_at == FIXED_NOW


# ------------------------------------------------------------------
# Politeness
# ------------------------------------------------------------------


def test_first_fetch_stores_the_body_and_validators(db):

    session = _session(
        _response(headers={"ETag": 'W/"v1"', "Last-Modified": "Wed, 05 Aug 2026 01:00:00 GMT"})
    )
    _make_source(db, session=session).fetch()

    cached = read_cache(db, URL)
    assert cached is not None
    assert cached.etag == 'W/"v1"'
    assert cached.last_modified == "Wed, 05 Aug 2026 01:00:00 GMT"
    assert cached.fetched_at == FIXED_NOW


def test_refetch_sends_conditional_headers(db):

    write_cache(
        db, URL, body=_ONE_EVENT, etag='W/"v1"',
        last_modified="Wed, 05 Aug 2026 01:00:00 GMT",
        fetched_at=FIXED_NOW - timedelta(hours=12),
    )
    session = _session()
    _make_source(db, session=session).fetch()

    headers = session.get.call_args.kwargs["headers"]
    assert headers["If-None-Match"] == 'W/"v1"'
    assert headers["If-Modified-Since"] == "Wed, 05 Aug 2026 01:00:00 GMT"


def test_not_modified_serves_the_cached_body(db):

    write_cache(
        db, URL, body=_ONE_EVENT, etag='W/"v1"', last_modified=None,
        fetched_at=FIXED_NOW - timedelta(hours=12),
    )
    session = _session(_response(body="", status=304))

    candidates = _make_source(db, session=session).fetch()

    assert len(candidates) == 1
    assert candidates[0].title == "Three Ply"


def test_within_the_fetch_interval_no_request_is_made(db):
    """Re-running the batch by hand must not mean re-downloading."""

    write_cache(
        db, URL, body=_ONE_EVENT, etag=None, last_modified=None,
        fetched_at=FIXED_NOW - timedelta(hours=1),
    )
    session = _session()

    candidates = _make_source(db, session=session).fetch()

    session.get.assert_not_called()
    assert len(candidates) == 1


def test_once_the_interval_elapses_it_refetches(db):

    write_cache(
        db, URL, body=_ONE_EVENT, etag=None, last_modified=None,
        fetched_at=FIXED_NOW - timedelta(hours=7),
    )
    session = _session()

    _make_source(db, session=session).fetch()

    session.get.assert_called_once()


def test_a_zero_interval_always_refetches(db):

    write_cache(
        db, URL, body=_ONE_EVENT, etag=None, last_modified=None, fetched_at=FIXED_NOW,
    )
    session = _session()

    _make_source(db, session=session, min_fetch_interval_hours=0.0).fetch()

    session.get.assert_called_once()


def test_request_identifies_itself_and_sets_a_timeout(db):

    session = _session()
    _make_source(db, session=session).fetch()

    kwargs = session.get.call_args.kwargs
    assert "what-do" in kwargs["headers"]["User-Agent"].lower()
    assert kwargs["timeout"] > 0


def test_http_error_propagates_without_retrying(db):

    response = _response()
    response.raise_for_status.side_effect = RuntimeError("503 Service Unavailable")
    session = _session(response)

    with pytest.raises(RuntimeError):
        _make_source(db, session=session).fetch()

    session.get.assert_called_once()
