"""Unit tests for the ICS (RFC 5545) parser."""

from __future__ import annotations

from datetime import datetime, timezone
import io

from src.ingestion.ics import parse_ics
from src.utils.logging import get_logger


def _logger():
    return get_logger("test", stream=io.StringIO())


def _calendar(*bodies: str) -> str:
    """Wrap VEVENT bodies in a VCALENDAR, using CRLF as the spec requires."""
    blocks = "".join(f"BEGIN:VEVENT\r\n{b}\r\nEND:VEVENT\r\n" for b in bodies)
    return (
        "BEGIN:VCALENDAR\r\n"
        "PRODID:-//Google Inc//Google Calendar 70.9054//EN\r\n"
        "VERSION:2.0\r\n"
        f"{blocks}"
        "END:VCALENDAR\r\n"
    )


def test_parses_a_single_event():

    ics = _calendar(
        "UID:abc123@google.com\r\n"
        "SUMMARY:General Trivia\r\n"
        "DTSTART:20260806T230000Z\r\n"
        "DTEND:20260807T010000Z\r\n"
        "STATUS:CONFIRMED"
    )

    events = parse_ics(ics, logger=_logger())

    assert len(events) == 1
    assert events[0].uid == "abc123@google.com"
    assert events[0].summary == "General Trivia"
    assert events[0].status == "CONFIRMED"


def test_parses_multiple_events():

    ics = _calendar(
        "UID:a@x\r\nSUMMARY:First\r\nDTSTART:20260806T230000Z",
        "UID:b@x\r\nSUMMARY:Second\r\nDTSTART:20260807T230000Z",
    )

    events = parse_ics(ics, logger=_logger())

    assert [e.summary for e in events] == ["First", "Second"]


def test_empty_calendar_yields_no_events():

    assert parse_ics(_calendar(), logger=_logger()) == []


def test_utc_timestamps_become_timezone_aware():

    ics = _calendar("UID:a@x\r\nDTSTART:20260806T230000Z\r\nDTEND:20260807T010000Z")

    event = parse_ics(ics, logger=_logger())[0]

    assert event.dtstart == datetime(2026, 8, 6, 23, 0, tzinfo=timezone.utc)
    assert event.dtend == datetime(2026, 8, 7, 1, 0, tzinfo=timezone.utc)


def test_missing_optional_fields_are_none():

    event = parse_ics(_calendar("UID:a@x\r\nSUMMARY:Bare"), logger=_logger())[0]

    assert event.dtend is None
    assert event.description is None
    assert event.location is None
    assert event.url is None


def test_folded_lines_are_unfolded():
    """RFC 5545 folds long values onto continuation lines beginning with a space."""

    ics = _calendar(
        "UID:a@x\r\n"
        "DESCRIPTION:Join us for classes on Wednesday evenings and Thursday\r\n"
        "  mornings. All you need is your willingness."
    )

    event = parse_ics(ics, logger=_logger())[0]

    assert event.description == (
        "Join us for classes on Wednesday evenings and Thursday mornings. "
        "All you need is your willingness."
    )


def test_escaped_characters_are_unescaped():

    ics = _calendar(
        "UID:a@x\r\n" "DESCRIPTION:Fees\\, times\\; and more\\nSecond line\\\\done"
    )

    event = parse_ics(ics, logger=_logger())[0]

    assert event.description == "Fees, times; and more\nSecond line\\done"


def test_escaped_comma_in_summary_survives():
    """Venue prefixes escape their commas; the adapter depends on getting them back."""

    ics = _calendar("UID:a@x\r\nSUMMARY:[The Rhumb Line\\, Gloucester\\, Music] Three Ply")

    event = parse_ics(ics, logger=_logger())[0]

    assert event.summary == "[The Rhumb Line, Gloucester, Music] Three Ply"


def test_non_ascii_survives_untouched():

    ics = _calendar("UID:a@x\r\nLOCATION:The Actors Studio – Crossroads Plaza")

    event = parse_ics(ics, logger=_logger())[0]

    assert event.location == "The Actors Studio – Crossroads Plaza"


def test_non_vevent_blocks_are_ignored():

    ics = (
        "BEGIN:VCALENDAR\r\n"
        "BEGIN:VTIMEZONE\r\n"
        "TZID:America/New_York\r\n"
        "BEGIN:DAYLIGHT\r\n"
        "TZNAME:EDT\r\n"
        "END:DAYLIGHT\r\n"
        "END:VTIMEZONE\r\n"
        "BEGIN:VEVENT\r\n"
        "UID:a@x\r\n"
        "SUMMARY:Real Event\r\n"
        "BEGIN:VALARM\r\n"
        "ACTION:DISPLAY\r\n"
        "END:VALARM\r\n"
        "END:VEVENT\r\n"
        "END:VCALENDAR\r\n"
    )

    events = parse_ics(ics, logger=_logger())

    assert len(events) == 1
    assert events[0].summary == "Real Event"


def test_bare_newline_line_endings_are_tolerated():
    """Not spec-compliant, but real feeds and fixtures get re-saved with LF."""

    ics = _calendar("UID:a@x\r\nSUMMARY:Trivia").replace("\r\n", "\n")

    events = parse_ics(ics, logger=_logger())

    assert len(events) == 1
    assert events[0].summary == "Trivia"


def test_unparseable_timestamp_does_not_kill_the_event():

    ics = _calendar("UID:a@x\r\nSUMMARY:Odd\r\nDTSTART:not-a-timestamp")

    events = parse_ics(ics, logger=_logger())

    assert len(events) == 1
    assert events[0].dtstart is None
    assert events[0].summary == "Odd"


def test_every_property_is_retained():
    """Lossless by construction: the extractor gets whatever the feed sent."""

    ics = _calendar("UID:a@x\r\nSUMMARY:Trivia\r\nTRANSP:OPAQUE\r\nSEQUENCE:3")

    event = parse_ics(ics, logger=_logger())[0]

    assert event.properties["TRANSP"] == "OPAQUE"
    assert event.properties["SEQUENCE"] == "3"
    assert event.properties["SUMMARY"] == "Trivia"


def test_url_property_is_captured_when_present():

    ics = _calendar("UID:a@x\r\nURL:https://example.com/event")

    assert parse_ics(ics, logger=_logger())[0].url == "https://example.com/event"


def test_all_day_event_parses_as_naive_midnight():
    """VALUE=DATE carries no time or zone; the caller decides how to localise."""

    ics = _calendar("UID:a@x\r\nDTSTART;VALUE=DATE:20260819")

    event = parse_ics(ics, logger=_logger())[0]

    assert event.dtstart == datetime(2026, 8, 19, 0, 0)
    assert event.dtstart.tzinfo is None


def test_tzid_timestamp_is_naive_and_reports_its_zone():
    """We do not silently pretend a zoned local time is UTC."""

    ics = _calendar("UID:a@x\r\nDTSTART;TZID=America/New_York:20260819T190000")

    event = parse_ics(ics, logger=_logger())[0]

    assert event.dtstart == datetime(2026, 8, 19, 19, 0)
    assert event.dtstart.tzinfo is None
    assert event.dtstart_tzid == "America/New_York"


def test_a_recurring_event_parses_without_warning():
    """The parser reports the rule; expanding it is `recurrence.expand`'s job.

    It used to warn that recurrences were not expanded. They are now, so the
    warning was not merely noisy — it was false, and one real feed emitted
    3,163 of them per fetch.
    """

    stream = io.StringIO()
    ics = _calendar(
        "UID:a@x\r\nSUMMARY:Weekly Trivia\r\n"
        "DTSTART:20260806T230000Z\r\nRRULE:FREQ=WEEKLY;COUNT=10"
    )

    events = parse_ics(ics, logger=get_logger("test", stream=stream))

    assert len(events) == 1
    assert events[0].repeated["RRULE"][0].value == "FREQ=WEEKLY;COUNT=10"
    assert stream.getvalue() == ""


def _body(*lines: str) -> str:
    """One VEVENT body, for `_calendar` to wrap."""
    return "\r\n".join(lines)


class TestRepeatedProperties:
    """A flat last-wins dict silently drops all but one EXDATE.

    Measured on a real feed: 405 events carry more than one EXDATE line and one
    carries 123. Keeping only the last would invent screenings that were
    cancelled.
    """

    def test_every_occurrence_of_a_repeated_property_is_kept(self):
        event = parse_ics(
            _calendar(
                _body(
                    "UID:e1",
                    "DTSTART:20260807T190000Z",
                    "EXDATE:20260814T190000Z",
                    "EXDATE:20260821T190000Z",
                    "EXDATE:20260828T190000Z",
                )
            )
        )[0]

        values = [p.value for p in event.repeated["EXDATE"]]

        assert values == ["20260814T190000Z", "20260821T190000Z", "20260828T190000Z"]

    def test_parameters_are_kept_alongside_each_value(self):
        """TZID appears on 2,029 of 2,035 real EXDATE lines; dropping it loses the zone."""
        event = parse_ics(
            _calendar(
                _body(
                    "UID:e1",
                    "DTSTART:20260807T190000Z",
                    "EXDATE;TZID=America/New_York:20260814T190000",
                )
            )
        )[0]

        exdate = event.repeated["EXDATE"][0]

        assert exdate.params["TZID"] == "America/New_York"

    def test_a_single_valued_property_is_also_available_repeated(self):
        event = parse_ics(_calendar(_body("UID:e1", "RRULE:FREQ=WEEKLY;BYDAY=TH")))[0]

        assert [p.value for p in event.repeated["RRULE"]] == ["FREQ=WEEKLY;BYDAY=TH"]

    def test_properties_still_holds_the_last_value(self):
        """The existing flat view is unchanged, so current callers keep working."""
        event = parse_ics(
            _calendar(_body("UID:e1", "EXDATE:20260814T190000Z", "EXDATE:20260821T190000Z"))
        )[0]

        assert event.properties["EXDATE"] == "20260821T190000Z"

    def test_repeated_properties_do_not_leak_between_events(self):
        events = parse_ics(
            _calendar(_body("UID:e1", "EXDATE:20260814T190000Z"), _body("UID:e2"))
        )

        assert "EXDATE" not in events[1].repeated

    def test_a_nested_block_does_not_contribute_repeated_properties(self):
        """VALARM carries its own TRIGGER; it must not land on the event."""
        event = parse_ics(
            _calendar(_body("UID:e1", "BEGIN:VALARM", "TRIGGER:-PT15M", "END:VALARM"))
        )[0]

        assert "TRIGGER" not in event.repeated
