"""Unit tests for the CLI's view filters."""

from datetime import date, datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest

from src.models.event import Event
from src.models.event_score import EventScore
from src.models.ranked_event import RankedEvent
from src.models.ranking import Ranking
from src.presentation.handles import short_handle
from src.presentation.filters import (
    matching,
    after_sunset,
    dated,
    during_night,
    night_of,
    on_date,
    overlapping,
    parse_time_window,
    undated,
)

TZ = timezone(timedelta(hours=-4))
TODAY = date(2025, 6, 21)
TOMORROW = date(2025, 6, 22)


def _pair(
    event_id: str = "evt-1",
    start: datetime | None = None,
    end: datetime | None = None,
    sunset: datetime | None = None,
    rank: int = 1,
    title: str | None = "Karaoke Night",
) -> RankedEvent:
    astro = {"sunset": sunset.isoformat()} if sunset is not None else None
    event = Event(
        event_id=event_id,
        source_event_candidates=[],
        source_type="instagram",
        created_at=datetime(2025, 6, 21, 9, 0),
        updated_at=datetime(2025, 6, 21, 9, 0),
        start_time=start,
        end_time=end,
        astronomical_data=astro,
        title=title,
    )
    return RankedEvent(
        event=event,
        score=EventScore(
            event_id=event_id, run_date=TODAY, base_score=0.4, match="yes"
        ),
        ranking=Ranking(
            event_id=event_id, run_date=TODAY, final_score=0.4, rank=rank
        ),
    )


def _at(hour: int, minute: int = 0, day: int = 21) -> datetime:
    return datetime(2025, 6, day, hour, minute, tzinfo=TZ)


def _ids(pairs) -> list[str]:
    return [ranked.event.event_id for ranked in pairs]


class TestParseTimeWindow:
    def test_parses_a_well_formed_window(self):
        assert parse_time_window("20:30-23:30") == (time(20, 30), time(23, 30))

    def test_tolerates_surrounding_whitespace(self):
        assert parse_time_window(" 20:30 - 23:30 ") == (time(20, 30), time(23, 30))

    @pytest.mark.parametrize(
        "spec", ["20:30", "", "-", "abc-def", "20:30-", "25:00-26:00", "20:30-23:30-01:00"]
    )
    def test_rejects_malformed_input(self, spec):
        with pytest.raises(ValueError):
            parse_time_window(spec)

    def test_rejects_a_window_that_crosses_midnight(self):
        """v1 does not wrap; erroring beats silently returning the inverse window."""
        with pytest.raises(ValueError):
            parse_time_window("23:00-01:00")

    def test_error_message_names_the_expected_format(self):
        with pytest.raises(ValueError, match="HH:MM-HH:MM"):
            parse_time_window("tonight")


class TestOnDate:
    def test_keeps_only_events_starting_that_day(self):
        pairs = [
            _pair("today", start=_at(20)),
            _pair("tomorrow", start=_at(20, day=22)),
        ]

        assert _ids(on_date(pairs, TODAY)) == ["today"]

    def test_drops_undated_events(self):
        """on_date is a claim about timing; undated events are surfaced separately."""
        assert on_date([_pair("undated", start=None)], TODAY) == []

    def test_preserves_input_order(self):
        pairs = [_pair("a", start=_at(22)), _pair("b", start=_at(19))]

        assert _ids(on_date(pairs, TODAY)) == ["a", "b"]


class TestDatedAndUndated:
    def test_undated_selects_events_with_no_start_time(self):
        pairs = [_pair("timed", start=_at(20)), _pair("untimed", start=None)]

        assert _ids(undated(pairs)) == ["untimed"]

    def test_dated_selects_events_with_a_start_time(self):
        pairs = [_pair("timed", start=_at(20)), _pair("untimed", start=None)]

        assert _ids(dated(pairs)) == ["timed"]

    def test_the_two_partition_the_input(self):
        pairs = [_pair("a", start=_at(20)), _pair("b", start=None), _pair("c", start=_at(21))]

        assert len(dated(pairs)) + len(undated(pairs)) == len(pairs)


class TestOverlapping:
    WINDOW = (time(20, 30), time(23, 30))

    def test_keeps_an_event_starting_inside_the_window(self):
        assert _ids(overlapping([_pair("in", start=_at(21))], *self.WINDOW, night=TODAY)) == ["in"]

    def test_drops_an_event_starting_before_the_window_with_no_end_time(self):
        assert overlapping([_pair("early", start=_at(19))], *self.WINDOW, night=TODAY) == []

    def test_drops_an_event_starting_after_the_window(self):
        assert overlapping([_pair("late", start=_at(23, 45))], *self.WINDOW, night=TODAY) == []

    def test_window_start_is_inclusive(self):
        assert _ids(overlapping([_pair("edge", start=_at(20, 30))], *self.WINDOW, night=TODAY)) == ["edge"]

    def test_window_end_is_inclusive(self):
        assert _ids(overlapping([_pair("edge", start=_at(23, 30))], *self.WINDOW, night=TODAY)) == ["edge"]

    def test_an_earlier_event_still_running_overlaps(self):
        """A gig from 19:00 to 22:00 is a real option at 20:30."""
        pairs = [_pair("running", start=_at(19), end=_at(22))]

        assert _ids(overlapping(pairs, *self.WINDOW, night=TODAY)) == ["running"]

    def test_an_event_ending_exactly_at_the_window_start_does_not_overlap(self):
        """Inverted 2026-08-14: the event's end is now exclusive.

        It used to count as overlapping, which is 82 of the 312 events a
        20:00 window wrongly kept. You cannot attend something that finishes as
        you arrive. The *window's* end stays inclusive — see the test below,
        where an event starting at the last minute is still an option.
        """
        pairs = [_pair("ending", start=_at(19), end=_at(20, 30))]

        assert overlapping(pairs, *self.WINDOW, night=TODAY) == []

    def test_an_event_ending_before_the_window_does_not(self):
        pairs = [_pair("done", start=_at(18), end=_at(20))]

        assert overlapping(pairs, *self.WINDOW, night=TODAY) == []

    def test_drops_undated_events(self):
        assert overlapping([_pair("undated", start=None)], *self.WINDOW, night=TODAY) == []

    def test_the_window_is_a_time_of_day_on_whichever_night_is_asked_about(self):
        """The window generalises across dates — but it is now anchored to the
        night in question rather than to the event's own start date.

        Anchoring to the event was the bug: for anything that began earlier the
        window was built on that earlier date, so a month-long exhibition
        cleared every window unconditionally. The intent survives; the caller
        says which night it means.
        """
        pairs = [_pair("tomorrow", start=_at(21, day=22))]

        assert _ids(overlapping(pairs, *self.WINDOW, night=TOMORROW)) == ["tomorrow"]
        assert overlapping(pairs, *self.WINDOW, night=TODAY) == []


class TestAfterSunset:
    def test_keeps_an_event_starting_after_sunset(self):
        pairs = [_pair("night", start=_at(21), sunset=_at(20, 15))]

        assert _ids(after_sunset(pairs)) == ["night"]

    def test_drops_an_event_starting_before_sunset(self):
        pairs = [_pair("day", start=_at(17), sunset=_at(20, 15))]

        assert after_sunset(pairs) == []

    def test_sunset_itself_does_not_count_as_after(self):
        pairs = [_pair("edge", start=_at(20, 15), sunset=_at(20, 15))]

        assert after_sunset(pairs) == []

    def test_drops_events_with_no_astronomical_data(self):
        """Without a sunset we cannot assert the event qualifies."""
        pairs = [_pair("unknown", start=_at(21), sunset=None)]

        assert after_sunset(pairs) == []

    def test_drops_undated_events(self):
        pairs = [_pair("undated", start=None, sunset=_at(20, 15))]

        assert after_sunset(pairs) == []

    def test_uses_each_events_own_sunset(self):
        """Sunset moves across dates; one shared value would misjudge the others."""
        pairs = [
            _pair("late-june", start=_at(20, 30), sunset=_at(20, 15)),
            _pair("shorter-day", start=_at(20, 30, day=22), sunset=_at(20, 45, day=22)),
        ]

        assert _ids(after_sunset(pairs)) == ["late-june"]


ZONE = ZoneInfo("America/New_York")


def _when(year: int, month: int, day: int, hour: int, minute: int = 0) -> datetime:
    """A wall-clock moment in the view's own zone."""
    return datetime(year, month, day, hour, minute, tzinfo=ZONE)


def test_night_of_before_rollover_is_the_previous_day():
    """00:30 belongs to the night still in progress, not the new calendar day."""
    assert night_of(_when(2025, 6, 21, 0, 30), time(4, 0)) == date(2025, 6, 20)


def test_night_of_at_the_rollover_starts_the_new_night():
    """The boundary is inclusive at the start, so 04:00 is already the new night."""
    assert night_of(_when(2025, 6, 21, 4, 0), time(4, 0)) == date(2025, 6, 21)


def test_night_of_after_the_rollover_is_the_same_day():
    assert night_of(_when(2025, 6, 21, 20, 0), time(4, 0)) == date(2025, 6, 21)


def test_night_of_at_midnight_rollover_is_the_calendar_date():
    """`00:00` restores plain calendar-day behaviour."""
    assert night_of(_when(2025, 6, 21, 0, 30), time(0, 0)) == date(2025, 6, 21)


def test_during_night_keeps_an_event_after_midnight():
    """The reason the window exists.

    A 00:30 show carries the *next* calendar date, so any filter keyed on the
    date alone loses it from the evening it actually belongs to.
    """
    late = _pair("late", start=_when(2025, 6, 22, 0, 30))

    assert during_night([late], date(2025, 6, 21), time(4, 0), ZONE) == [late]


def test_during_night_excludes_the_previous_night():
    early = _pair("early", start=_when(2025, 6, 21, 3, 0))

    assert during_night([early], date(2025, 6, 21), time(4, 0), ZONE) == []


def test_during_night_is_half_open_at_both_ends():
    """Start inclusive, end exclusive, so consecutive nights cannot both claim one event."""
    at_start = _pair("at-start", start=_when(2025, 6, 21, 4, 0))
    at_end = _pair("at-end", start=_when(2025, 6, 22, 4, 0))

    kept = during_night([at_start, at_end], date(2025, 6, 21), time(4, 0), ZONE)

    assert kept == [at_start]


def test_during_night_localises_a_naive_start_time():
    """A naive start must not crash the whole view.

    Normalization guarantees aware datetimes, but comparing naive to aware
    raises TypeError, and one bad row taking down every event is the worse
    failure. Assume the view's zone, which is what normalization would have.
    """
    naive = _pair("naive", start=datetime(2025, 6, 21, 20, 0))

    assert during_night([naive], date(2025, 6, 21), time(4, 0), ZONE) == [naive]


def test_during_night_excludes_undated_events():
    """Undated events are selected by `undated()` and rendered separately."""
    assert during_night([_pair("no-time")], date(2025, 6, 21), time(4, 0), ZONE) == []


def test_during_night_compares_across_zones():
    """An event stored in another offset is judged by the view's wall clock."""
    utc_evening = _pair("utc", start=datetime(2025, 6, 22, 1, 0, tzinfo=timezone.utc))

    assert during_night([utc_evening], date(2025, 6, 21), time(4, 0), ZONE) == [utc_evening]


def test_during_night_preserves_order():
    """Filters may drop a pair, never move one — rank order is the product."""
    first = _pair("first", start=_when(2025, 6, 21, 22, 0))
    second = _pair("second", start=_when(2025, 6, 21, 19, 0))

    assert during_night([first, second], date(2025, 6, 21), time(4, 0), ZONE) == [first, second]


class TestMultiNightEvents:
    """An exhibition open all month is on every night, not just its first.

    It stays one stored event — the CLI shows it once per night it is actually
    open, rather than the batch minting thirty copies.
    """

    def _run(self, start, end, night):
        pair = _pair("run", start=start, end=end)
        return during_night([pair], night, time(4, 0), ZONE) == [pair]

    def test_shown_on_its_opening_night(self):
        assert self._run(_when(2026, 8, 1, 10), _when(2026, 8, 30, 18), date(2026, 8, 1))

    def test_shown_on_a_night_in_the_middle(self):
        assert self._run(_when(2026, 8, 1, 10), _when(2026, 8, 30, 18), date(2026, 8, 15))

    def test_shown_on_its_final_night(self):
        assert self._run(_when(2026, 8, 1, 10), _when(2026, 8, 30, 18), date(2026, 8, 30))

    def test_not_shown_the_night_after_it_closes(self):
        assert not self._run(_when(2026, 8, 1, 10), _when(2026, 8, 30, 18), date(2026, 8, 31))

    def test_not_shown_the_night_before_it_opens(self):
        assert not self._run(_when(2026, 8, 5, 10), _when(2026, 8, 30, 18), date(2026, 8, 4))

    def test_an_evening_event_is_unaffected(self):
        assert self._run(_when(2026, 8, 7, 19), _when(2026, 8, 7, 23), date(2026, 8, 7))
        assert not self._run(_when(2026, 8, 7, 19), _when(2026, 8, 7, 23), date(2026, 8, 8))

    def test_an_event_with_no_end_is_still_instantaneous(self):
        assert self._run(_when(2026, 8, 7, 19), None, date(2026, 8, 7))
        assert not self._run(_when(2026, 8, 7, 19), None, date(2026, 8, 8))

    def test_a_run_appears_once_per_night_not_once_per_day_of_its_length(self):
        """One row in, one row out — the CLI never multiplies an event."""
        pair = _pair("run", start=_when(2026, 8, 1, 10), end=_when(2026, 8, 30, 18))

        assert during_night([pair], date(2026, 8, 15), time(4, 0), ZONE) == [pair]


class TestMatching:
    """How `--explain` names an event.

    Not by rank. Displayed numbering is view-local, so the number beside an
    event names a different event in every view, and a reader who types the
    number they counted would get a valid explanation of the wrong event —
    silently. The two selectors are a `#handle`, which is stable and visibly
    not a position, and a title substring.
    """

    def test_a_handle_selects_its_event(self):
        pairs = [_pair("a", title="First"), _pair("b", title="Second")]

        selector = f"#{short_handle('b')}"

        assert [p.event.title for p in matching(pairs, selector)] == ["Second"]

    def test_a_handle_prefix_is_enough(self):
        """As git does. Any length resolves, because ambiguity is answered by
        returning every match rather than by demanding more characters."""
        pairs = [_pair("a", title="First"), _pair("b", title="Second")]

        selector = f"#{short_handle('b')[:4]}"

        assert [p.event.title for p in matching(pairs, selector)] == ["Second"]

    def test_an_ambiguous_handle_prefix_returns_every_match(self):
        """A collision becomes likely past ~10k events. It must degrade to a
        question, never to a wrong answer."""
        pairs = [_pair("a", title="First"), _pair("b", title="Second")]

        assert len(matching(pairs, "#")) == 2

    def test_a_handle_that_matches_nothing_matches_nothing(self):
        assert matching([_pair("a")], "#ffffffff") == []

    def test_the_handle_is_not_reachable_without_its_sigil(self):
        """The sigil is what makes a handle unmistakable for a position. Bare,
        an all-digit handle would be exactly the ambiguity being removed."""
        pairs = [_pair("a", title="First")]

        assert matching(pairs, short_handle("a")) == []

    def test_text_selects_by_title_substring(self):
        pairs = [_pair("a", title="Karaoke Night"), _pair("b", title="Open Mic")]

        assert [p.event.title for p in matching(pairs, "karaoke")] == ["Karaoke Night"]

    def test_the_substring_ignores_case(self):
        assert len(matching([_pair("a", title="Karaoke Night")], "KARAOKE")) == 1

    def test_an_ambiguous_substring_returns_every_match(self):
        """The caller lists them rather than guessing. Picking one silently is
        how a person ends up reading the wrong event's explanation."""
        pairs = [
            _pair("a", title="Karaoke Night"),
            _pair("b", title="Karaoke Brunch"),
        ]

        assert len(matching(pairs, "karaoke")) == 2

    def test_a_bare_number_is_a_title_substring(self):
        """With the integer path gone, a numeric title is reachable at last —
        `--explain 1984` finds the film. Previously the integer test came first
        and was exclusive, so no title containing digits could be named."""
        pairs = [_pair("a", rank=7, title="1984"), _pair("b", rank=1984, title="Quiz")]

        assert [p.event.title for p in matching(pairs, "1984")] == ["1984"]

    def test_a_rank_is_no_longer_a_selector(self):
        """The behaviour deliberately removed. A number names titles now."""
        pairs = [_pair("a", rank=1, title="First"), _pair("b", rank=2, title="Second")]

        assert matching(pairs, "2") == []

    def test_an_untitled_event_is_not_matched_by_text(self):
        assert matching([_pair("a", title=None)], "anything") == []


class TestOverlappingCorrectness:
    """`--time` answers "is this on during these hours", and it was wrong.

    Measured 2026-08-14 against the live ranking: **312** events survive a
    20:00–23:59 window despite starting before it. 207 genuinely overlap and are
    an arrival-semantics question nothing stored can answer yet. The other 105
    are these two bugs.
    """

    def test_the_window_is_anchored_to_the_night_not_the_event(self):
        """A month-long exhibition that opened on the 1st built its window on the
        1st, so its end date cleared it unconditionally — the comparison stopped
        meaning anything for any event that began earlier."""
        exhibition = _pair(
            "e", start=_at(9, day=1), end=_at(17, day=30), sunset=None
        )

        kept = overlapping([exhibition], time(20, 0), time(23, 59), night=date(2025, 6, 21))

        assert kept == []

    def test_an_event_ending_exactly_as_the_window_opens_is_excluded(self):
        """82 of the 105. You cannot attend something that has just finished."""
        finished = _pair("e", start=_at(18), end=_at(20))

        kept = overlapping([finished], time(20, 0), time(23, 59), night=TODAY)

        assert kept == []

    def test_an_event_starting_exactly_as_the_window_closes_is_kept(self):
        """The other end stays inclusive — you can still arrive."""
        late = _pair("e", start=_at(23, 59), end=_at(23, 59) + timedelta(hours=1))

        kept = overlapping([late], time(20, 0), time(23, 59), night=TODAY)

        assert len(kept) == 1

    def test_a_long_span_is_read_as_a_daily_programme(self):
        """A workshop stored as Aug 10 09:00 -> Aug 14 12:00 is 96 unbroken hours
        and overlapped every window that could be typed. Its endpoints are its
        daily hours, and 09:00–12:00 is not an evening."""
        workshop = _pair("e", start=_at(9, day=10), end=_at(12, day=14))

        kept = overlapping([workshop], time(20, 0), time(23, 59), night=date(2025, 6, 12))

        assert kept == []

    def test_a_long_span_whose_daily_hours_do_match_is_kept(self):
        """The rule must not simply drop everything long. A week-long festival
        running 20:00–23:00 nightly is exactly what the window is asking for."""
        festival = _pair("e", start=_at(20, day=10), end=_at(23, day=14))

        kept = overlapping([festival], time(20, 0), time(23, 59), night=date(2025, 6, 12))

        assert len(kept) == 1

    def test_a_whole_number_of_days_is_genuinely_continuous(self):
        """A mooring rental stored 12:00 -> 12:00 next day is not a programme
        that runs at noon; you have the mooring at 8pm. Equal times-of-day mean
        the span is whole days, which is the one readable signal of continuity."""
        mooring = _pair("e", start=_at(12, day=10), end=_at(12, day=11))

        kept = overlapping([mooring], time(20, 0), time(23, 59), night=date(2025, 6, 10))

        assert len(kept) == 1

    def test_a_short_evening_event_is_unaffected(self):
        """The ordinary case, which none of this may disturb."""
        gig = _pair("e", start=_at(21), end=_at(23))

        assert len(overlapping([gig], time(20, 0), time(23, 59), night=TODAY)) == 1


class TestConfigurableSpanThreshold:
    """`long_span_hours` decides when a span stops being one occurrence.

    The only one of the view's four numbers with no per-invocation override, and
    the one most likely to want tuning against real listings — it is a claim
    about how sources publish, not a display preference (#31).
    """

    def test_the_threshold_is_taken_from_the_caller(self):
        """A 30-hour span is one occurrence at the default and a daily
        programme at 24 — so the same event answers differently, which is the
        whole reason for making it configurable."""
        span = _pair("e", start=_at(9, day=10), end=_at(15, day=11))

        as_continuous = overlapping(
            [span], time(20, 0), time(23, 59), night=date(2025, 6, 10), long_span_hours=48
        )
        as_programme = overlapping(
            [span], time(20, 0), time(23, 59), night=date(2025, 6, 10), long_span_hours=24
        )

        assert len(as_continuous) == 1
        assert as_programme == []
