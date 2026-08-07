"""Unit tests for the shared night boundary.

Ingestion and the CLI must agree on which day it is. When they disagree, a
re-run late in the evening discards the events the CLI is still showing.
"""

from datetime import date, datetime, time
from zoneinfo import ZoneInfo

from src.utils.nights import night_of, night_start

ZONE = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")


class TestNightOf:
    def test_before_the_rollover_is_the_previous_day(self):
        assert night_of(datetime(2026, 8, 8, 0, 30, tzinfo=ZONE), time(4, 0)) == date(2026, 8, 7)

    def test_at_the_rollover_the_new_night_begins(self):
        assert night_of(datetime(2026, 8, 8, 4, 0, tzinfo=ZONE), time(4, 0)) == date(2026, 8, 8)

    def test_after_the_rollover_is_the_same_day(self):
        assert night_of(datetime(2026, 8, 8, 20, 0, tzinfo=ZONE), time(4, 0)) == date(2026, 8, 8)

    def test_a_midnight_rollover_is_the_calendar_date(self):
        assert night_of(datetime(2026, 8, 8, 0, 30, tzinfo=ZONE), time(0, 0)) == date(2026, 8, 8)


class TestNightStart:
    def test_the_current_night_began_at_the_rollover(self):
        now = datetime(2026, 8, 7, 20, 0, tzinfo=ZONE)

        assert night_start(now, time(4, 0), ZONE) == datetime(2026, 8, 7, 4, 0, tzinfo=ZONE)

    def test_after_midnight_the_night_began_yesterday(self):
        """The floor must not advance past events still happening."""
        now = datetime(2026, 8, 8, 0, 30, tzinfo=ZONE)

        assert night_start(now, time(4, 0), ZONE) == datetime(2026, 8, 7, 4, 0, tzinfo=ZONE)

    def test_a_moment_in_another_zone_is_converted_first(self):
        """The batch clock may be UTC while the location is not."""
        now = datetime(2026, 8, 8, 2, 0, tzinfo=UTC)  # 22:00 on the 7th in New York

        assert night_start(now, time(4, 0), ZONE) == datetime(2026, 8, 7, 4, 0, tzinfo=ZONE)

    def test_the_floor_never_lands_in_the_future(self):
        for hour in range(24):
            now = datetime(2026, 8, 7, hour, 30, tzinfo=ZONE)
            assert night_start(now, time(4, 0), ZONE) <= now
