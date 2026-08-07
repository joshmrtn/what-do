"""Unit tests for RRULE expansion.

Pure: a VEvent and a window in, occurrence start times out. No I/O, no clock of
its own.
"""

from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from src.ingestion.ics import Property, VEvent
from src.ingestion.recurrence import expand, expand_calendar

ZONE = ZoneInfo("America/New_York")
UTC = timezone.utc

#: A 30-day horizon, the shape `scraping.horizon_days` gives ingestion.
WINDOW_START = datetime(2026, 8, 7, tzinfo=ZONE)
WINDOW_END = WINDOW_START + timedelta(days=30)


class _Logger:
    def __init__(self):
        self.warnings: list[str] = []

    def warning(self, message, **kwargs):
        self.warnings.append(message)

    def info(self, message, **kwargs):
        pass


def _vevent(
    dtstart: datetime | None = None,
    rrule: str | None = None,
    exdates: list[tuple[str, dict[str, str]]] | None = None,
    dtstart_tzid: str | None = None,
) -> VEvent:
    repeated: dict[str, list[Property]] = {}
    if rrule is not None:
        repeated["RRULE"] = [Property(value=rrule)]
    if exdates:
        repeated["EXDATE"] = [Property(value=v, params=p) for v, p in exdates]
    return VEvent(uid="e1", dtstart=dtstart, dtstart_tzid=dtstart_tzid, repeated=repeated)


class TestExpansion:
    def test_a_long_running_weekly_rule_yields_occurrences_in_the_window(self):
        """The case the window alone would have destroyed.

        Karaoke every Thursday since 2023 has a base occurrence two years back,
        so filtering on that date drops a live weekly event.
        """
        event = _vevent(datetime(2023, 1, 5, 20, 0, tzinfo=ZONE), rrule="FREQ=WEEKLY;BYDAY=TH")

        occurrences = expand(event, WINDOW_START, WINDOW_END, ZONE)

        assert occurrences == [
            datetime(2026, 8, 13, 20, 0, tzinfo=ZONE),
            datetime(2026, 8, 20, 20, 0, tzinfo=ZONE),
            datetime(2026, 8, 27, 20, 0, tzinfo=ZONE),
            datetime(2026, 9, 3, 20, 0, tzinfo=ZONE),
        ]

    def test_an_unbounded_rule_terminates(self):
        """9 real rules have neither UNTIL nor COUNT; the window is what stops them."""
        event = _vevent(datetime(2010, 1, 1, 19, 0, tzinfo=ZONE), rrule="FREQ=DAILY")

        occurrences = expand(event, WINDOW_START, WINDOW_END, ZONE)

        assert len(occurrences) == 30

    def test_a_rule_that_already_ended_yields_nothing(self):
        """2,971 of 2,972 real DAILY rules are expired film runs."""
        event = _vevent(datetime(2022, 3, 1, 19, 0, tzinfo=ZONE), rrule="FREQ=DAILY;COUNT=7")

        assert expand(event, WINDOW_START, WINDOW_END, ZONE) == []

    def test_the_window_is_half_open(self):
        """Start inclusive, end exclusive, so two windows cannot both claim one."""
        event = _vevent(WINDOW_START, rrule="FREQ=DAILY")

        occurrences = expand(event, WINDOW_START, WINDOW_END, ZONE)

        assert occurrences[0] == WINDOW_START
        assert WINDOW_END not in occurrences

    def test_interval_is_honoured(self):
        event = _vevent(datetime(2026, 8, 7, 19, 0, tzinfo=ZONE), rrule="FREQ=DAILY;INTERVAL=10")

        occurrences = expand(event, WINDOW_START, WINDOW_END, ZONE)

        assert occurrences == [
            datetime(2026, 8, 7, 19, 0, tzinfo=ZONE),
            datetime(2026, 8, 17, 19, 0, tzinfo=ZONE),
            datetime(2026, 8, 27, 19, 0, tzinfo=ZONE),
        ]


class TestExclusions:
    def test_an_excluded_date_is_removed(self):
        event = _vevent(
            datetime(2026, 8, 6, 19, 0, tzinfo=ZONE),
            rrule="FREQ=WEEKLY;BYDAY=TH",
            exdates=[("20260813T190000", {"TZID": "America/New_York"})],
        )

        occurrences = expand(event, WINDOW_START, WINDOW_END, ZONE)

        assert datetime(2026, 8, 13, 19, 0, tzinfo=ZONE) not in occurrences
        assert datetime(2026, 8, 20, 19, 0, tzinfo=ZONE) in occurrences

    def test_every_excluded_date_is_removed(self):
        """405 real events carry more than one EXDATE; one carries 123."""
        event = _vevent(
            datetime(2026, 8, 6, 19, 0, tzinfo=ZONE),
            rrule="FREQ=WEEKLY;BYDAY=TH",
            exdates=[
                ("20260813T190000", {"TZID": "America/New_York"}),
                ("20260820T190000", {"TZID": "America/New_York"}),
                ("20260827T190000", {"TZID": "America/New_York"}),
            ],
        )

        occurrences = expand(event, WINDOW_START, WINDOW_END, ZONE)

        assert occurrences == [datetime(2026, 9, 3, 19, 0, tzinfo=ZONE)]

    def test_a_utc_exclusion_matches_a_zoned_occurrence(self):
        """The same instant written two ways must still cancel."""
        event = _vevent(
            datetime(2026, 8, 6, 19, 0, tzinfo=ZONE),
            rrule="FREQ=WEEKLY;BYDAY=TH",
            exdates=[("20260813T230000Z", {})],
        )

        assert datetime(2026, 8, 13, 19, 0, tzinfo=ZONE) not in expand(
            event, WINDOW_START, WINDOW_END, ZONE
        )

    def test_an_unparseable_exclusion_is_ignored_not_fatal(self):
        logger = _Logger()
        event = _vevent(
            datetime(2026, 8, 6, 19, 0, tzinfo=ZONE),
            rrule="FREQ=WEEKLY;BYDAY=TH",
            exdates=[("not-a-date", {})],
        )

        occurrences = expand(event, WINDOW_START, WINDOW_END, ZONE, logger=logger)

        assert len(occurrences) == 4
        assert logger.warnings


class TestNonRecurring:
    def test_a_plain_event_yields_its_own_start(self):
        """One path for every event, so callers do not branch on RRULE."""
        start = datetime(2026, 8, 10, 19, 0, tzinfo=ZONE)

        assert expand(_vevent(start), WINDOW_START, WINDOW_END, ZONE) == [start]

    def test_a_plain_event_outside_the_window_yields_nothing(self):
        assert expand(_vevent(datetime(2020, 1, 1, tzinfo=ZONE)), WINDOW_START, WINDOW_END, ZONE) == []

    def test_an_event_with_no_start_yields_nothing(self):
        """Nothing can be placed on a clock, so nothing can be judged against a window."""
        assert expand(_vevent(None), WINDOW_START, WINDOW_END, ZONE) == []


class TestGracefulDegradation:
    """An unknown construct must degrade to the base occurrence, never a drop.

    Falling back keeps the event; guessing dates would invent them. This is the
    property that makes a hand-rolled parser acceptable alongside dateutil.
    """

    def test_an_unparseable_rule_falls_back_to_the_base_occurrence(self):
        logger = _Logger()
        start = datetime(2026, 8, 10, 19, 0, tzinfo=ZONE)
        event = _vevent(start, rrule="FREQ=NONSENSE;WAT=1")

        occurrences = expand(event, WINDOW_START, WINDOW_END, ZONE, logger=logger)

        assert occurrences == [start]
        assert logger.warnings

    def test_a_fallback_outside_the_window_still_yields_nothing(self):
        """Degrading must not smuggle an expired event past the window."""
        event = _vevent(datetime(2019, 5, 1, 19, 0, tzinfo=ZONE), rrule="FREQ=NONSENSE")

        assert expand(event, WINDOW_START, WINDOW_END, ZONE) == []

    def test_an_empty_rule_falls_back(self):
        start = datetime(2026, 8, 10, 19, 0, tzinfo=ZONE)

        assert expand(_vevent(start, rrule=""), WINDOW_START, WINDOW_END, ZONE) == [start]

    def test_degrading_without_a_logger_does_not_raise(self):
        start = datetime(2026, 8, 10, 19, 0, tzinfo=ZONE)

        assert expand(_vevent(start, rrule="FREQ=NONSENSE"), WINDOW_START, WINDOW_END, ZONE) == [start]


class TestNaiveTimestamps:
    """`VEvent` keeps zoned values naive and reports the zone separately."""

    def test_a_naive_start_uses_its_declared_zone(self):
        event = _vevent(
            datetime(2026, 8, 10, 19, 0),
            rrule="FREQ=DAILY;COUNT=1",
            dtstart_tzid="America/New_York",
        )

        assert expand(event, WINDOW_START, WINDOW_END, ZONE) == [
            datetime(2026, 8, 10, 19, 0, tzinfo=ZONE)
        ]

    def test_a_naive_start_without_a_zone_uses_the_fallback(self):
        event = _vevent(datetime(2026, 8, 10, 19, 0), rrule="FREQ=DAILY;COUNT=1")

        assert expand(event, WINDOW_START, WINDOW_END, ZONE) == [
            datetime(2026, 8, 10, 19, 0, tzinfo=ZONE)
        ]

    def test_an_unknown_zone_falls_back_rather_than_raising(self):
        event = _vevent(
            datetime(2026, 8, 10, 19, 0),
            rrule="FREQ=DAILY;COUNT=1",
            dtstart_tzid="Mars/Olympus_Mons",
        )

        assert expand(event, WINDOW_START, WINDOW_END, ZONE) == [
            datetime(2026, 8, 10, 19, 0, tzinfo=ZONE)
        ]


class TestDaylightSaving:
    def test_a_weekly_rule_holds_its_wall_clock_across_the_dst_boundary(self):
        """20:00 stays 20:00 after the clocks change, rather than drifting to 19:00."""
        event = _vevent(datetime(2026, 10, 29, 20, 0, tzinfo=ZONE), rrule="FREQ=WEEKLY;BYDAY=TH")
        window_start = datetime(2026, 10, 29, tzinfo=ZONE)

        occurrences = expand(event, window_start, window_start + timedelta(days=21), ZONE)

        assert [o.hour for o in occurrences] == [20, 20, 20]


class TestNonConformantUntil:
    """RFC 5545 requires UNTIL in UTC when DTSTART is zoned; feeds ignore this.

    A real feed writes `UNTIL=20210208` — a bare date — on 50 zoned rules.
    dateutil enforces the rule strictly, so without normalising, a *future*
    rule written this way would collapse to its base occurrence and silently
    lose every later date.
    """

    def test_a_bare_date_until_still_expands(self):
        event = _vevent(
            datetime(2026, 8, 6, 19, 0, tzinfo=ZONE), rrule="FREQ=WEEKLY;UNTIL=20260821"
        )

        occurrences = expand(event, WINDOW_START, WINDOW_END, ZONE)

        assert occurrences == [
            datetime(2026, 8, 13, 19, 0, tzinfo=ZONE),
            datetime(2026, 8, 20, 19, 0, tzinfo=ZONE),
        ]

    def test_a_bare_date_until_includes_the_whole_final_day(self):
        """A date-only UNTIL names a day, so an evening event on it still counts."""
        event = _vevent(
            datetime(2026, 8, 13, 19, 0, tzinfo=ZONE), rrule="FREQ=WEEKLY;UNTIL=20260820"
        )

        assert datetime(2026, 8, 20, 19, 0, tzinfo=ZONE) in expand(
            event, WINDOW_START, WINDOW_END, ZONE
        )

    def test_a_naive_datetime_until_still_expands(self):
        event = _vevent(
            datetime(2026, 8, 6, 19, 0, tzinfo=ZONE),
            rrule="FREQ=WEEKLY;UNTIL=20260821T000000",
        )

        assert len(expand(event, WINDOW_START, WINDOW_END, ZONE)) == 2

    def test_a_utc_until_is_left_alone(self):
        event = _vevent(
            datetime(2026, 8, 6, 19, 0, tzinfo=ZONE),
            rrule="FREQ=WEEKLY;UNTIL=20260821T000000Z",
        )

        assert len(expand(event, WINDOW_START, WINDOW_END, ZONE)) == 2

    def test_normalising_does_not_warn(self):
        """A feed writing UNTIL the common way is not a degradation."""
        logger = _Logger()
        event = _vevent(
            datetime(2026, 8, 6, 19, 0, tzinfo=ZONE), rrule="FREQ=WEEKLY;UNTIL=20260821"
        )

        expand(event, WINDOW_START, WINDOW_END, ZONE, logger=logger)

        assert logger.warnings == []


def _override(
    uid: str,
    recurrence_id: str,
    dtstart: datetime,
    summary: str = "Moved",
    status: str | None = None,
    tzid: str = "America/New_York",
) -> VEvent:
    """A VEVENT that replaces one occurrence of its series."""
    return VEvent(
        uid=uid,
        summary=summary,
        status=status,
        dtstart=dtstart,
        repeated={"RECURRENCE-ID": [Property(value=recurrence_id, params={"TZID": tzid})]},
    )


def _series(uid: str, dtstart: datetime, rrule: str, summary: str = "Weekly") -> VEvent:
    return VEvent(
        uid=uid, summary=summary, dtstart=dtstart, repeated={"RRULE": [Property(value=rrule)]}
    )


class TestRecurrenceOverrides:
    """2,357 real events carry a RECURRENCE-ID.

    Ignoring them double-books a moved showing at both its old and new slot.
    """

    def test_a_moved_occurrence_uses_its_new_time(self):
        master = _series("s1", datetime(2026, 8, 6, 19, 0, tzinfo=ZONE), "FREQ=WEEKLY;BYDAY=TH")
        moved = _override("s1", "20260813T190000", datetime(2026, 8, 13, 21, 30, tzinfo=ZONE))

        starts = [o.start for o in expand_calendar([master, moved], WINDOW_START, WINDOW_END, ZONE)]

        assert datetime(2026, 8, 13, 21, 30, tzinfo=ZONE) in starts
        assert datetime(2026, 8, 13, 19, 0, tzinfo=ZONE) not in starts

    def test_a_moved_occurrence_is_not_also_emitted_at_its_old_slot(self):
        master = _series("s1", datetime(2026, 8, 6, 19, 0, tzinfo=ZONE), "FREQ=WEEKLY;BYDAY=TH")
        moved = _override("s1", "20260813T190000", datetime(2026, 8, 13, 21, 30, tzinfo=ZONE))

        occurrences = expand_calendar([master, moved], WINDOW_START, WINDOW_END, ZONE)

        assert len(occurrences) == 4

    def test_the_override_supplies_the_fields_for_its_occurrence(self):
        master = _series("s1", datetime(2026, 8, 6, 19, 0, tzinfo=ZONE), "FREQ=WEEKLY;BYDAY=TH")
        moved = _override(
            "s1", "20260813T190000", datetime(2026, 8, 13, 21, 30, tzinfo=ZONE), summary="Special"
        )

        by_start = {o.start: o for o in expand_calendar([master, moved], WINDOW_START, WINDOW_END, ZONE)}

        assert by_start[datetime(2026, 8, 13, 21, 30, tzinfo=ZONE)].event.summary == "Special"
        assert by_start[datetime(2026, 8, 20, 19, 0, tzinfo=ZONE)].event.summary == "Weekly"

    def test_identity_follows_the_original_slot_not_the_new_time(self):
        """A moved instance is still the same instance, so it keeps its slot.

        Keying on the new time would orphan the stored event every time the
        showing moved again.
        """
        master = _series("s1", datetime(2026, 8, 6, 19, 0, tzinfo=ZONE), "FREQ=WEEKLY;BYDAY=TH")
        moved = _override("s1", "20260813T190000", datetime(2026, 8, 13, 21, 30, tzinfo=ZONE))

        by_start = {o.start: o for o in expand_calendar([master, moved], WINDOW_START, WINDOW_END, ZONE)}

        assert by_start[datetime(2026, 8, 13, 21, 30, tzinfo=ZONE)].original_start == datetime(
            2026, 8, 13, 19, 0, tzinfo=ZONE
        )

    def test_a_cancelled_override_removes_the_occurrence(self):
        master = _series("s1", datetime(2026, 8, 6, 19, 0, tzinfo=ZONE), "FREQ=WEEKLY;BYDAY=TH")
        cancelled = _override(
            "s1", "20260813T190000", datetime(2026, 8, 13, 19, 0, tzinfo=ZONE), status="CANCELLED"
        )

        starts = [o.start for o in expand_calendar([master, cancelled], WINDOW_START, WINDOW_END, ZONE)]

        assert datetime(2026, 8, 13, 19, 0, tzinfo=ZONE) not in starts
        assert len(starts) == 3

    def test_an_override_can_move_an_occurrence_out_of_the_window(self):
        master = _series("s1", datetime(2026, 8, 6, 19, 0, tzinfo=ZONE), "FREQ=WEEKLY;BYDAY=TH")
        pushed = _override("s1", "20260813T190000", datetime(2027, 1, 5, 19, 0, tzinfo=ZONE))

        starts = [o.start for o in expand_calendar([master, pushed], WINDOW_START, WINDOW_END, ZONE)]

        assert len(starts) == 3

    def test_an_override_can_move_an_occurrence_into_the_window(self):
        """The replaced slot is outside; the new time is inside."""
        master = _series("s1", datetime(2026, 6, 4, 19, 0, tzinfo=ZONE), "FREQ=WEEKLY;BYDAY=TH;UNTIL=20260701T000000Z")
        pulled = _override("s1", "20260611T190000", datetime(2026, 8, 12, 19, 0, tzinfo=ZONE))

        starts = [o.start for o in expand_calendar([master, pulled], WINDOW_START, WINDOW_END, ZONE)]

        assert starts == [datetime(2026, 8, 12, 19, 0, tzinfo=ZONE)]

    def test_an_orphan_override_still_yields_its_occurrence(self):
        """A feed can ship an override whose master is outside the document."""
        orphan = _override("gone", "20260813T190000", datetime(2026, 8, 13, 21, 0, tzinfo=ZONE))

        starts = [o.start for o in expand_calendar([orphan], WINDOW_START, WINDOW_END, ZONE)]

        assert starts == [datetime(2026, 8, 13, 21, 0, tzinfo=ZONE)]

    def test_overrides_do_not_leak_between_series(self):
        first = _series("s1", datetime(2026, 8, 6, 19, 0, tzinfo=ZONE), "FREQ=WEEKLY;BYDAY=TH")
        second = _series("s2", datetime(2026, 8, 6, 19, 0, tzinfo=ZONE), "FREQ=WEEKLY;BYDAY=TH", summary="Other")
        moved = _override("s1", "20260813T190000", datetime(2026, 8, 13, 21, 30, tzinfo=ZONE))

        occurrences = expand_calendar([first, second, moved], WINDOW_START, WINDOW_END, ZONE)
        others = [o for o in occurrences if o.event.summary == "Other"]

        assert len(others) == 4

    def test_a_plain_event_reports_itself_as_its_own_slot(self):
        start = datetime(2026, 8, 10, 19, 0, tzinfo=ZONE)
        plain = VEvent(uid="p1", dtstart=start)

        occurrence = expand_calendar([plain], WINDOW_START, WINDOW_END, ZONE)[0]

        assert occurrence.start == occurrence.original_start == start

    def test_an_event_without_a_uid_is_still_expanded(self):
        """No UID means no series to join, not a reason to lose the event."""
        start = datetime(2026, 8, 10, 19, 0, tzinfo=ZONE)

        assert len(expand_calendar([VEvent(dtstart=start)], WINDOW_START, WINDOW_END, ZONE)) == 1

    def test_occurrences_come_back_in_chronological_order(self):
        master = _series("s1", datetime(2026, 8, 6, 19, 0, tzinfo=ZONE), "FREQ=WEEKLY;BYDAY=TH")
        moved = _override("s1", "20260813T190000", datetime(2026, 8, 11, 21, 30, tzinfo=ZONE))

        starts = [o.start for o in expand_calendar([master, moved], WINDOW_START, WINDOW_END, ZONE)]

        assert starts == sorted(starts)
