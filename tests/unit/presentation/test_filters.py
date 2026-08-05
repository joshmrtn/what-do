"""Unit tests for the CLI's view filters."""

from datetime import date, datetime, time, timedelta, timezone

import pytest

from src.models.event import Event
from src.models.recommendation import Recommendation
from src.presentation.filters import (
    after_sunset,
    dated,
    on_date,
    overlapping,
    parse_time_window,
    undated,
)

TZ = timezone(timedelta(hours=-4))
TODAY = date(2025, 6, 21)


def _pair(
    event_id: str = "evt-1",
    start: datetime | None = None,
    end: datetime | None = None,
    sunset: datetime | None = None,
) -> tuple[Recommendation, Event]:
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
    )
    recommendation = Recommendation(
        recommendation_id=f"{TODAY.isoformat()}:{event_id}",
        event_id=event_id,
        run_date=TODAY,
        base_score=0.4,
        weather_adjustment=0.0,
        tag_confidence=1.0,
        final_score=0.4,
        match="yes",
        tier="top_pick",
        rank=1,
    )
    return recommendation, event


def _at(hour: int, minute: int = 0, day: int = 21) -> datetime:
    return datetime(2025, 6, day, hour, minute, tzinfo=TZ)


def _ids(pairs) -> list[str]:
    return [e.event_id for _, e in pairs]


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
        assert _ids(overlapping([_pair("in", start=_at(21))], *self.WINDOW)) == ["in"]

    def test_drops_an_event_starting_before_the_window_with_no_end_time(self):
        assert overlapping([_pair("early", start=_at(19))], *self.WINDOW) == []

    def test_drops_an_event_starting_after_the_window(self):
        assert overlapping([_pair("late", start=_at(23, 45))], *self.WINDOW) == []

    def test_window_start_is_inclusive(self):
        assert _ids(overlapping([_pair("edge", start=_at(20, 30))], *self.WINDOW)) == ["edge"]

    def test_window_end_is_inclusive(self):
        assert _ids(overlapping([_pair("edge", start=_at(23, 30))], *self.WINDOW)) == ["edge"]

    def test_an_earlier_event_still_running_overlaps(self):
        """A gig from 19:00 to 22:00 is a real option at 20:30."""
        pairs = [_pair("running", start=_at(19), end=_at(22))]

        assert _ids(overlapping(pairs, *self.WINDOW)) == ["running"]

    def test_an_event_ending_exactly_at_the_window_start_overlaps(self):
        pairs = [_pair("ending", start=_at(19), end=_at(20, 30))]

        assert _ids(overlapping(pairs, *self.WINDOW)) == ["ending"]

    def test_an_event_ending_before_the_window_does_not(self):
        pairs = [_pair("done", start=_at(18), end=_at(20))]

        assert overlapping(pairs, *self.WINDOW) == []

    def test_drops_undated_events(self):
        assert overlapping([_pair("undated", start=None)], *self.WINDOW) == []

    def test_compares_against_each_events_own_date(self):
        """The window is a time of day, so tomorrow's 21:00 event matches too."""
        pairs = [_pair("tomorrow", start=_at(21, day=22))]

        assert _ids(overlapping(pairs, *self.WINDOW)) == ["tomorrow"]


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
