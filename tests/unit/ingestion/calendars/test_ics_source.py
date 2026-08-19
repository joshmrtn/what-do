"""Unit tests for IcsCalendarSource."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from unittest.mock import MagicMock
import io

import pytest

from src.storage.memory.http_cache import InMemoryHttpCache
from src.config import FeedConfig
from src.ingestion.candidate_id import derive_content_id
from src.ingestion.calendars.ics_source import IcsCalendarSource
from src.models.event_candidate import EventCandidate
from src.storage.sqlite.connection import init_db
from src.utils.logging import get_logger
from tests.support.network import fetcher_for

FIXED_NOW = datetime(2026, 8, 5, 2, 0, tzinfo=timezone.utc)
URL = "https://calendar.example.com/public/basic.ics"
EASTERN = ZoneInfo("America/New_York")


@pytest.fixture
def cache():
    """No database: these tests are about conditional requests."""
    return InMemoryHttpCache()


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


def _make_source(cache, session=None, now=FIXED_NOW, stream=None, horizon_days=30,
                 timezone_name="UTC", content_ids=False, **config_overrides):
    settings = {
        "name": "northshorenightout",
        "url": URL,
        "source_type": "northshorenightout",
        "min_fetch_interval_hours": 6.0,
    }
    settings.update(config_overrides)

    return IcsCalendarSource(
        config=FeedConfig(**settings),
        fetcher=fetcher_for(session or _session(), urls=URL, http_cache=cache, now=now),
        get_now=lambda: now,
        logger=get_logger("test", stream=stream or io.StringIO()),
        horizon_days=horizon_days,
        timezone_name=timezone_name,
        uses_content_id=lambda source: content_ids,
    )


def test_returns_event_candidates(cache):

    results = _make_source(cache).fetch()

    assert len(results) == 1
    assert isinstance(results[0], EventCandidate)


def test_id_derives_from_uid_and_is_stable_across_fetches(cache):
    """A nightly refetch must update its rows, not duplicate them."""

    first = _make_source(cache).fetch()[0]
    later = _make_source(cache, now=FIXED_NOW + timedelta(days=1)).fetch()[0]

    assert first.id == later.id
    assert "abc123@google.com" in first.id


class TestWhenThePublishersUidsAreNotTrusted:
    """`northshorenightout` is a Google Calendar export that re-mints every UID
    between days — 860 stored rows for 156 listings. A source latched to content
    ids keys on the listing instead, and must collapse exactly the rows the
    churn detector counted as one.
    """

    def _renamed_uid(self, uid: str) -> str:
        return _calendar(
            f"UID:{uid}\r\n"
            "SUMMARY:​[The Rhumb Line\\, Gloucester\\, Music] Three Ply\r\n"
            "DTSTART:20260809T010000Z\r\n"
            "DTEND:20260809T040000Z\r\n"
            "STATUS:CONFIRMED"
        )

    def test_a_re_minted_uid_yields_the_same_id(self, cache):
        """The defect this exists for. Same listing, new handle on it."""
        first = _make_source(
            cache, session=_session(_response(self._renamed_uid("first@google.com"))),
            content_ids=True,
        ).fetch()[0]
        later = _make_source(
            cache, session=_session(_response(self._renamed_uid("second@google.com"))),
            now=FIXED_NOW + timedelta(days=1), content_ids=True,
        ).fetch()[0]

        assert first.id == later.id

    def test_the_uid_is_not_in_the_id_at_all(self, cache):
        first = _make_source(cache, content_ids=True).fetch()[0]

        assert "abc123@google.com" not in first.id

    def test_a_re_cased_republish_yields_the_same_id(self, cache):
        """Canonical on both text fields, or the latch fires and the duplicates
        survive it — a venue writing its listings in caps was invisible to dedup
        for exactly this reason."""
        shouted = _calendar(
            "UID:abc123@google.com\r\n"
            "SUMMARY:​[THE RHUMB LINE\\, Gloucester\\, Music] THREE PLY\r\n"
            "DTSTART:20260809T010000Z\r\n"
            "DTEND:20260809T040000Z\r\n"
            "STATUS:CONFIRMED"
        )

        normal = _make_source(cache, content_ids=True).fetch()[0]
        # A day on, or the conditional cache serves the first body back and the
        # republish is never read — the assertion would hold vacuously.
        loud = _make_source(
            cache, session=_session(_response(shouted)), content_ids=True,
            now=FIXED_NOW + timedelta(days=1),
        ).fetch()[0]

        assert normal.id == loud.id

    def test_two_occurrences_of_a_series_stay_distinct(self, cache):
        """A recurring programme shares one UID *and* one title and venue across
        every date it runs, so the start is what tells tonight from next week.
        Without it a whole season collapses into one candidate."""
        series = _calendar(
            "UID:weekly@google.com\r\n"
            "SUMMARY:​[The Rhumb Line\\, Gloucester\\, Music] Open Mic\r\n"
            "DTSTART:20260809T010000Z\r\n"
            "DTEND:20260809T040000Z\r\n"
            "RRULE:FREQ=WEEKLY;COUNT=3\r\n"
            "STATUS:CONFIRMED"
        )

        results = _make_source(
            cache, session=_session(_response(series)), content_ids=True
        ).fetch()

        assert len(results) == 3
        assert len({c.id for c in results}) == 3

    def test_an_all_day_listing_is_keyed_on_the_start_it_stores(self, cache):
        """`_place_start` moves an all-day event off midnight and onto its own
        night, and `__post_init__` canonicalises to UTC. An id derived before
        either step describes a value the row never holds, so the listing is
        re-minted on every fetch — measured live, one all-day listing came back
        as a brand new row the first fetch after the re-key."""
        all_day = _calendar(
            "UID:allday@google.com\r\n"
            "SUMMARY:​[The Rhumb Line\\, Gloucester\\, Music] Auditions\r\n"
            "DTSTART;VALUE=DATE:20260810\r\n"
            "DTEND;VALUE=DATE:20260811\r\n"
            "STATUS:CONFIRMED"
        )

        candidate = _make_source(
            cache, session=_session(_response(all_day)), content_ids=True
        ).fetch()[0]

        assert candidate.id == derive_content_id(
            source=candidate.source,
            title=candidate.title,
            venue=candidate.venue,
            start=candidate.start_time,
        )

    def test_every_candidate_is_keyed_on_its_own_stored_fields(self, cache):
        """The invariant, stated once: the id is a function of what the row
        holds. It is the same thing the re-key verifies before committing."""
        for candidate in _make_source(cache, content_ids=True).fetch():
            assert candidate.id == derive_content_id(
                source=candidate.source,
                title=candidate.title,
                venue=candidate.venue,
                start=candidate.start_time,
            )

    def test_a_different_listing_still_gets_a_different_id(self, cache):
        other = _calendar(
            "UID:abc123@google.com\r\n"
            "SUMMARY:​[The Rhumb Line\\, Gloucester\\, Music] Someone Else\r\n"
            "DTSTART:20260809T010000Z\r\n"
            "DTEND:20260809T040000Z\r\n"
            "STATUS:CONFIRMED"
        )

        first = _make_source(cache, content_ids=True).fetch()[0]
        second = _make_source(
            cache, session=_session(_response(other)), content_ids=True,
            now=FIXED_NOW + timedelta(days=1),
        ).fetch()[0]

        assert first.id != second.id


def test_source_and_source_type_come_from_config(cache):

    candidate = _make_source(cache, source_type="community_calendar").fetch()[0]

    assert candidate.source == "northshorenightout"
    assert candidate.source_type == "community_calendar"


def test_venue_and_city_parse_out_of_the_title_prefix(cache):

    candidate = _make_source(cache).fetch()[0]

    assert candidate.title == "Three Ply"
    assert candidate.venue == "The Rhumb Line"
    assert candidate.location == "Gloucester"


def test_zero_width_space_is_stripped_from_the_title(cache):

    candidate = _make_source(cache).fetch()[0]

    assert "​" not in candidate.title
    assert "​" not in candidate.venue


def test_a_carried_category_becomes_structured_metadata(cache):
    """Not prose. The prompt names `Event category` and says what it is — a
    taxonomy from a listing site, not a claim about this event — and it can only
    do that for a field it can see."""

    candidate = _make_source(cache).fetch()[0]

    assert candidate.metadata["listing_category"] == "Music"
    assert candidate.description is None


def test_two_segment_prefix_has_no_category(cache):

    ics = _calendar(
        "UID:a@x\r\nSUMMARY:​[Bent Water Brewing CO.\\, Lynn] General Trivia\r\n"
        "DTSTART:20260806T230000Z"
    )
    candidate = _make_source(cache, session=_session(_response(ics))).fetch()[0]

    assert candidate.title == "General Trivia"
    assert candidate.venue == "Bent Water Brewing CO."
    assert candidate.location == "Lynn"
    assert candidate.description is None


def test_the_feeds_own_description_is_left_alone(cache):
    """It used to be welded to the heading with a blank line that normalization
    then collapsed, so the model read `Category: Music Doors at 7.`."""

    ics = _calendar(
        "UID:a@x\r\nSUMMARY:​[Venue\\, City\\, Music] Show\r\n"
        "DESCRIPTION:Doors at 7.\r\nDTSTART:20260806T230000Z"
    )
    candidate = _make_source(cache, session=_session(_response(ics))).fetch()[0]

    assert candidate.description == "Doors at 7."
    assert candidate.metadata["listing_category"] == "Music"


def test_a_null_bucket_heading_is_dropped_entirely(cache):
    """Measured on the live database: 30 stored events carried `Other` and 21
    carried `Karaoke & trivia` into the model as unlabelled prose — the exact
    headings the HTML path already withheld."""

    for heading in ("Other", "Karaoke & trivia"):
        ics = _calendar(
            f"UID:a@x\r\nSUMMARY:​[Venue\\, City\\, {heading}] Trivia\r\n"
            "DTSTART:20260806T230000Z"
        )
        candidate = _make_source(cache, session=_session(_response(ics))).fetch()[0]

        assert "listing_category" not in candidate.metadata, heading
        assert candidate.description is None, heading


def test_a_dropped_heading_still_leaves_the_description(cache):
    ics = _calendar(
        "UID:a@x\r\nSUMMARY:​[Venue\\, City\\, Other] Show\r\n"
        "DESCRIPTION:Doors at 7.\r\nDTSTART:20260806T230000Z"
    )
    candidate = _make_source(cache, session=_session(_response(ics))).fetch()[0]

    assert candidate.description == "Doors at 7."
    assert "listing_category" not in candidate.metadata


def test_unconventional_summary_keeps_the_event_and_warns(cache):
    """A convention change must cost venue attribution, never the event.

    The warning counts rather than naming: one feed does this on all 12,038 of
    its events, so quoting each title buries the rest of the run.
    """

    stream = io.StringIO()
    ics = _calendar("UID:a@x\r\nSUMMARY:Just A Title\r\nDTSTART:20260806T230000Z")

    candidates = _make_source(
        cache, session=_session(_response(ics)), stream=stream
    ).fetch()

    assert len(candidates) == 1
    assert candidates[0].title == "Just A Title"
    assert candidates[0].venue is None
    assert "1 of 1 summaries" in stream.getvalue()


def test_timestamps_map_to_start_and_end(cache):

    candidate = _make_source(cache).fetch()[0]

    assert candidate.start_time == datetime(2026, 8, 9, 1, 0, tzinfo=timezone.utc)
    assert candidate.end_time == datetime(2026, 8, 9, 4, 0, tzinfo=timezone.utc)


def test_raw_published_at_is_never_set(cache):
    """CREATED tracks calendar rebuilds, and the field drives the lookback discard."""

    ics = _calendar(
        "UID:a@x\r\nSUMMARY:​[V\\, C] Show\r\n"
        "CREATED:20200101T000000Z\r\nDTSTART:20260806T230000Z"
    )
    candidate = _make_source(cache, session=_session(_response(ics))).fetch()[0]

    assert candidate.raw_published_at is None


def test_cancelled_events_are_skipped(cache):

    ics = _calendar(
        "UID:a@x\r\nSUMMARY:​[V\\, C] Gone\r\n"
        "DTSTART:20260806T230000Z\r\nSTATUS:CANCELLED",
        "UID:b@x\r\nSUMMARY:​[V\\, C] Still On\r\nDTSTART:20260806T230000Z",
    )
    candidates = _make_source(cache, session=_session(_response(ics))).fetch()

    assert [c.title for c in candidates] == ["Still On"]


def test_html_in_the_description_is_converted_not_dropped(cache):

    ics = _calendar(
        "UID:a@x\r\nSUMMARY:​[V\\, C] Show\r\n"
        "DESCRIPTION:Boston!<br>You will<br>see <a href=\"https://x.test/t\">tickets</a>\r\n"
        "DTSTART:20260806T230000Z"
    )
    candidate = _make_source(cache, session=_session(_response(ics))).fetch()[0]

    assert "Boston!\nYou will" in candidate.description
    assert "https://x.test/t" in candidate.description


def test_url_property_becomes_the_candidate_url(cache):

    ics = _calendar(
        "UID:a@x\r\nSUMMARY:​[V\\, C] Show\r\n"
        "URL:https://example.com/event\r\nDTSTART:20260806T230000Z"
    )
    candidate = _make_source(cache, session=_session(_response(ics))).fetch()[0]

    assert candidate.url == "https://example.com/event"


def test_event_without_a_uid_is_skipped(cache):
    """Without a stable key the candidate would duplicate on every run."""

    ics = _calendar("SUMMARY:​[V\\, C] Anonymous\r\nDTSTART:20260806T230000Z")

    assert _make_source(cache, session=_session(_response(ics))).fetch() == []


def test_discovered_at_uses_the_injected_clock(cache):

    candidate = _make_source(cache).fetch()[0]

    assert candidate.discovered_at == FIXED_NOW


# ------------------------------------------------------------------
# Politeness
# ------------------------------------------------------------------


def test_first_fetch_stores_the_body_and_validators(cache):

    session = _session(
        _response(headers={"ETag": 'W/"v1"', "Last-Modified": "Wed, 05 Aug 2026 01:00:00 GMT"})
    )
    _make_source(cache, session=session).fetch()

    cached = cache.get(URL)
    assert cached is not None
    assert cached.etag == 'W/"v1"'
    assert cached.last_modified == "Wed, 05 Aug 2026 01:00:00 GMT"
    assert cached.fetched_at == FIXED_NOW


def test_refetch_sends_conditional_headers(cache):

    cache.put(
        URL, body=_ONE_EVENT, etag='W/"v1"',
        last_modified="Wed, 05 Aug 2026 01:00:00 GMT",
        fetched_at=FIXED_NOW - timedelta(hours=12),
    )
    session = _session()
    _make_source(cache, session=session).fetch()

    headers = session.get.call_args.kwargs["headers"]
    assert headers["If-None-Match"] == 'W/"v1"'
    assert headers["If-Modified-Since"] == "Wed, 05 Aug 2026 01:00:00 GMT"


def test_not_modified_serves_the_cached_body(cache):

    cache.put(
        URL, body=_ONE_EVENT, etag='W/"v1"', last_modified=None,
        fetched_at=FIXED_NOW - timedelta(hours=12),
    )
    session = _session(_response(body="", status=304))

    candidates = _make_source(cache, session=session).fetch()

    assert len(candidates) == 1
    assert candidates[0].title == "Three Ply"


def test_within_the_fetch_interval_no_request_is_made(cache):
    """Re-running the batch by hand must not mean re-downloading."""

    cache.put(
        URL, body=_ONE_EVENT, etag=None, last_modified=None,
        fetched_at=FIXED_NOW - timedelta(hours=1),
    )
    session = _session()

    candidates = _make_source(cache, session=session).fetch()

    session.get.assert_not_called()
    assert len(candidates) == 1


def test_once_the_interval_elapses_it_refetches(cache):

    cache.put(
        URL, body=_ONE_EVENT, etag=None, last_modified=None,
        fetched_at=FIXED_NOW - timedelta(hours=7),
    )
    session = _session()

    _make_source(cache, session=session).fetch()

    session.get.assert_called_once()


def test_a_zero_interval_always_refetches(cache):

    cache.put(
        URL, body=_ONE_EVENT, etag=None, last_modified=None, fetched_at=FIXED_NOW,
    )
    session = _session()

    _make_source(cache, session=session, min_fetch_interval_hours=0.0).fetch()

    session.get.assert_called_once()


def test_request_identifies_itself_and_sets_a_timeout(cache):

    session = _session()
    _make_source(cache, session=session).fetch()

    kwargs = session.get.call_args.kwargs
    assert "what-do" in kwargs["headers"]["User-Agent"].lower()
    assert kwargs["timeout"] > 0


def test_http_error_propagates_without_retrying(cache):

    response = _response()
    response.raise_for_status.side_effect = RuntimeError("503 Service Unavailable")
    session = _session(response)

    with pytest.raises(RuntimeError):
        _make_source(cache, session=session).fetch()

    session.get.assert_called_once()


_WEEKLY = _calendar(
    "UID:weekly@google.com\r\n"
    "SUMMARY:[The Rhumb Line\\, Gloucester\\, Music] Karaoke\r\n"
    "DTSTART;TZID=America/New_York:20230105T200000\r\n"
    "DTEND;TZID=America/New_York:20230105T230000\r\n"
    "RRULE:FREQ=WEEKLY;BYDAY=TH"
)


def _weekly_source(cache, now=FIXED_NOW, horizon_days=30, body=_WEEKLY, stream=None,
                   timezone_name="UTC", **config_overrides):
    return _make_source(cache, session=_session(_response(body)), now=now, stream=stream,
                        horizon_days=horizon_days, timezone_name=timezone_name,
                        **config_overrides)


class TestRecurringEvents:
    """A weekly event running since 2023 must yield this month's occurrences.

    Its base occurrence is two years old, so without expansion the adapter emits
    one candidate dated 2023 and every live Thursday is lost.
    """

    def test_a_recurring_event_yields_one_candidate_per_occurrence(self, cache):
        results = _weekly_source(cache).fetch()

        assert len(results) == 4

    def test_each_occurrence_lands_on_its_own_date(self, cache):
        """Localised before the date is taken, and that is not a formality: a
        candidate holds its instant in UTC, so an 8pm Eastern occurrence is
        already the next day in UTC and `.date()` alone reports it a day late.
        The same trap `_scope_filter` avoids, and the reason no production code
        derives a date from a candidate."""
        starts = sorted(c.start_time for c in _weekly_source(cache).fetch())

        # The horizon runs from the start of tonight, so 30 days is 30 whole
        # nights: the 3 September occurrence is night 31.
        assert [s.astimezone(EASTERN).date().isoformat() for s in starts] == [
            "2026-08-06", "2026-08-13", "2026-08-20", "2026-08-27"
        ]

    def test_every_occurrence_gets_a_distinct_id(self, cache):
        results = _weekly_source(cache).fetch()

        assert len({c.id for c in results}) == len(results)

    def test_an_occurrence_keeps_its_id_across_fetches(self, cache):
        """The whole cost model rests on this.

        A stable id lets reconcile match tonight's occurrence to the event
        stored last night, so the extraction hash skips it. Get it wrong and
        every occurrence re-extracts nightly at ~3 min each.
        """
        first = {c.id for c in _weekly_source(cache).fetch()}
        later = {c.id for c in _weekly_source(cache, now=FIXED_NOW + timedelta(hours=12)).fetch()}

        assert first & later

    def test_the_occurrence_id_carries_the_uid_and_its_slot(self, cache):
        results = sorted(_weekly_source(cache).fetch(), key=lambda c: c.start_time)

        assert "weekly@google.com" in results[0].id
        assert results[0].id != results[1].id

    def test_occurrences_beyond_the_horizon_are_not_emitted(self, cache):
        results = _weekly_source(cache, horizon_days=7).fetch()

        assert len(results) == 1

    def test_the_shared_fields_carry_to_every_occurrence(self, cache):
        results = _weekly_source(cache).fetch()

        assert all(c.title == "Karaoke" for c in results)
        assert all(c.venue == "The Rhumb Line" for c in results)

    def test_a_non_recurring_event_keeps_a_uid_only_id(self, cache):
        """Nothing disambiguates a one-off, so its id should not churn."""
        candidate = _make_source(cache).fetch()[0]

        assert candidate.id == "northshorenightout:abc123@google.com"

    def test_an_expired_series_yields_nothing(self, cache):
        expired = _calendar(
            "UID:old@google.com\r\n"
            "SUMMARY:[A Venue\\, Salem] Old Run\r\n"
            "DTSTART;TZID=America/New_York:20220301T190000\r\n"
            "RRULE:FREQ=DAILY;COUNT=7"
        )

        assert _weekly_source(cache, body=expired).fetch() == []

    def test_an_excluded_occurrence_is_not_emitted(self, cache):
        with_exclusion = _calendar(
            "UID:weekly@google.com\r\n"
            "SUMMARY:[A Venue\\, Salem] Karaoke\r\n"
            "DTSTART;TZID=America/New_York:20230105T200000\r\n"
            "RRULE:FREQ=WEEKLY;BYDAY=TH\r\n"
            "EXDATE;TZID=America/New_York:20260813T200000"
        )

        dates = {
            c.start_time.astimezone(EASTERN).date().isoformat()
            for c in _weekly_source(cache, body=with_exclusion).fetch()
        }

        assert "2026-08-13" not in dates
        assert "2026-08-20" in dates

    def test_a_moved_occurrence_uses_the_override_fields(self, cache):
        with_override = _calendar(
            "UID:weekly@google.com\r\n"
            "SUMMARY:[A Venue\\, Salem] Karaoke\r\n"
            "DTSTART;TZID=America/New_York:20230105T200000\r\n"
            "RRULE:FREQ=WEEKLY;BYDAY=TH",
            "UID:weekly@google.com\r\n"
            "SUMMARY:[A Venue\\, Salem] Special Guest\r\n"
            "DTSTART;TZID=America/New_York:20260813T213000\r\n"
            "RECURRENCE-ID;TZID=America/New_York:20260813T200000",
        )

        results = _weekly_source(cache, body=with_override).fetch()
        titles = {c.title for c in results}

        assert "Special Guest" in titles
        assert len(results) == 4


class TestNoLowerBoundOnAnnouncementAge:
    """The only time bound is the event's own, never how long ago it was announced.

    A concert announced four months ahead is exactly the kind of thing worth
    knowing about — often the kind you must book early. The window is on event
    time in both directions: nothing is too old to have been *announced*, and
    anything already over is gone.
    """

    def test_a_long_announced_event_is_still_ingested(self, cache):
        long_announced = _calendar(
            "UID:early@google.com\r\n"
            "SUMMARY:[The Cabot\\, Beverly\\, Music] Booked Months Ago\r\n"
            "CREATED:20260101T000000Z\r\n"
            "DTSTAMP:20260101T000000Z\r\n"
            "LAST-MODIFIED:20260101T000000Z\r\n"
            "DTSTART:20260810T230000Z"
        )

        results = _weekly_source(cache, body=long_announced).fetch()

        assert len(results) == 1
        assert results[0].title == "Booked Months Ago"

    def test_an_event_that_has_already_happened_is_dropped(self, cache):
        """A previous night is of no interest, however recently it was announced.

        Dated for the night now in progress it would be kept — an event three
        hours ago has not stopped being tonight's.
        """
        yesterday = _calendar(
            "UID:gone@google.com\r\n"
            "SUMMARY:[A Venue\\, Salem] Two Nights Ago\r\n"
            "DTSTART:20260802T230000Z"
        )

        assert _weekly_source(cache, body=yesterday).fetch() == []

    def test_announcement_age_never_reaches_the_lookback(self, cache):
        """`raw_published_at` stays unset, which is what the lookback discards on.

        Setting it from CREATED would make a forward-looking calendar expire
        against a rule written for stale social posts.
        """
        long_announced = _calendar(
            "UID:early@google.com\r\n"
            "SUMMARY:[A Venue\\, Salem] Booked Months Ago\r\n"
            "CREATED:20260101T000000Z\r\n"
            "DTSTART:20260810T230000Z"
        )

        assert _weekly_source(cache, body=long_announced).fetch()[0].raw_published_at is None


class TestWindowFloorIsTheNight:
    """A re-run must not discard the evening it is being run during.

    Flooring the window at `now` means an 20:00 re-run drops a 19:00 event that
    has not finished, and a 00:30 re-run drops the whole evening the CLI is
    still showing.
    """

    _TONIGHT = _calendar(
        "UID:tonight@google.com\r\n"
        "SUMMARY:[A Venue\\, Salem] Doors At Seven\r\n"
        "DTSTART;TZID=America/New_York:20260807T190000"
    )

    def _at(self, cache, now):
        """The night boundary is local, so these must run in the venue's zone."""
        return _weekly_source(
            cache, now=now, body=self._TONIGHT, timezone_name="America/New_York"
        ).fetch()

    def test_the_overnight_batch_ingests_tonight(self, cache):
        assert self._at(cache, datetime(2026, 8, 7, 2, 0, tzinfo=EASTERN))

    def test_a_re_run_during_the_event_still_ingests_it(self, cache):
        assert self._at(cache, datetime(2026, 8, 7, 20, 0, tzinfo=EASTERN))

    def test_a_re_run_after_midnight_still_ingests_that_evening(self, cache):
        assert self._at(cache, datetime(2026, 8, 8, 0, 30, tzinfo=EASTERN))

    def test_the_previous_night_is_still_dropped(self, cache):
        """Dated 8/7 is of no interest once 8/8 has begun."""
        assert self._at(cache, datetime(2026, 8, 8, 10, 0, tzinfo=EASTERN)) == []


_BARE_TITLE = _calendar(
    "UID:film@google.com\r\n"
    "SUMMARY:BROOKLYN\r\n"
    "DTSTART:20260809T010000Z"
)

_WITH_PREFIX = _calendar(
    "UID:gig@google.com\r\n"
    "SUMMARY:[The Rhumb Line\\, Gloucester] Three Ply\r\n"
    "DTSTART:20260809T010000Z"
)


class TestFeedVenueDefaults:
    """A cinema's own calendar names the film, not the venue.

    Every event then arrives venue-less, which costs blocklist matching, dedup,
    and the CLI's `Title — Venue` line.
    """

    def test_the_feed_venue_fills_in_when_the_summary_declares_none(self, cache):
        candidate = _weekly_source(
            cache, body=_BARE_TITLE, venue="Cape Ann Community Cinema"
        ).fetch()[0]

        assert candidate.venue == "Cape Ann Community Cinema"
        assert candidate.title == "BROOKLYN"

    def test_the_feed_city_fills_in_too(self, cache):
        candidate = _weekly_source(cache, body=_BARE_TITLE, city="Gloucester").fetch()[0]

        assert candidate.location == "Gloucester"

    def test_a_declared_venue_wins_over_the_feed_default(self, cache):
        """The summary is more specific, so an aggregator's attribution stands."""
        candidate = _weekly_source(
            cache, body=_WITH_PREFIX, venue="Wrong Venue", city="Wrong City"
        ).fetch()[0]

        assert candidate.venue == "The Rhumb Line"
        assert candidate.location == "Gloucester"

    def test_without_a_default_the_venue_stays_none(self, cache):
        candidate = _weekly_source(cache, body=_BARE_TITLE).fetch()[0]

        assert candidate.venue is None

    def test_the_default_reaches_every_occurrence_of_a_series(self, cache):
        recurring = _calendar(
            "UID:weekly@google.com\r\n"
            "SUMMARY:Karaoke\r\n"
            "DTSTART;TZID=America/New_York:20230105T200000\r\n"
            "RRULE:FREQ=WEEKLY;BYDAY=TH"
        )

        results = _weekly_source(cache, body=recurring, venue="The Rhumb Line").fetch()

        assert results
        assert all(c.venue == "The Rhumb Line" for c in results)


class TestSummaryWarningVolume:
    """One real feed names the film and nothing else, on all 12,038 events.

    Warning per event buries every other line in the run.
    """

    _THREE_BARE = _calendar(
        "UID:a@x\r\nSUMMARY:BROOKLYN\r\nDTSTART:20260809T010000Z",
        "UID:b@x\r\nSUMMARY:Restrepo\r\nDTSTART:20260809T020000Z",
        "UID:c@x\r\nSUMMARY:Leaving\r\nDTSTART:20260809T030000Z",
    )

    def _warnings(self, cache, **overrides):
        stream = io.StringIO()
        _weekly_source(cache, body=self._THREE_BARE, stream=stream, **overrides).fetch()
        return [ln for ln in stream.getvalue().splitlines() if "convention" in ln]

    def test_a_feed_with_a_venue_does_not_warn_at_all(self, cache):
        """Nothing is missing: the feed declared its venue, so there is no gap."""
        assert self._warnings(cache, venue="Cape Ann Community Cinema") == []

    def test_without_a_venue_the_warning_is_summarised_not_repeated(self, cache):
        warnings = self._warnings(cache)

        assert len(warnings) == 1

    def test_the_summary_reports_how_many(self, cache):
        assert "3" in self._warnings(cache)[0]

    def test_a_conforming_feed_stays_silent(self, cache):
        stream = io.StringIO()
        _weekly_source(cache, body=_WITH_PREFIX, stream=stream).fetch()

        assert "convention" not in stream.getvalue()
