"""Tests for whether a read-time rescore should run at all.

The decision is deliberately separate from the recomputation. Most invocations
answer "no" and must cost nothing, and the two "no"s that matter — a batch is
running, and the ranking is not for tonight — are about not making things worse
rather than about the forecast.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

from src.presentation.rescore_policy import (
    batch_is_running,
    BATCH_RUNNING,
    FORECAST_FRESH,
    NOT_TONIGHT,
    RESCORE,
    should_rescore,
)

TONIGHT = date(2026, 8, 17)
NOW = datetime(2026, 8, 17, 20, 0, tzinfo=timezone.utc)
TTL = timedelta(minutes=60)


def _decide(
    *,
    run_date: date = TONIGHT,
    forecast_issued_at: datetime | None = NOW - timedelta(hours=14),
    batch_running: bool = False,
) -> str:
    return should_rescore(
        run_date=run_date,
        tonight=TONIGHT,
        forecast_issued_at=forecast_issued_at,
        now=NOW,
        ttl=TTL,
        batch_running=batch_running,
    )


def test_a_stale_forecast_on_tonights_run_is_rescored():
    assert _decide() == RESCORE


def test_a_fresh_forecast_is_left_alone():
    """The second invocation seconds later must cost nothing."""
    assert _decide(forecast_issued_at=NOW - timedelta(minutes=5)) == FORECAST_FRESH


def test_a_listing_with_no_forecast_is_left_alone():
    """Nothing to refresh. An all-indoor night is not a stale one."""
    assert _decide(forecast_issued_at=None) == FORECAST_FRESH


def test_a_running_batch_stops_the_rescore():
    """`busy_timeout` means the CLI would *wait*, which is the worst outcome.

    A batch takes eight hours and writes throughout. Waiting on its lock turns
    a listing that should be instant into one that hangs, and writing beside it
    is what the standing rule against disturbing a running process forbids.
    """
    assert _decide(batch_running=True) == BATCH_RUNNING


def test_a_running_batch_wins_over_a_stale_forecast():
    """Ordering matters: the reason not to write outranks the reason to."""
    assert (
        _decide(batch_running=True, forecast_issued_at=NOW - timedelta(days=2))
        == BATCH_RUNNING
    )


def test_a_ranking_from_an_earlier_night_is_not_rescored():
    """The scope floor is derived from the run date.

    Rescoring a day-old run would apply tonight's forecast under a scope
    anchored to yesterday — mixing the two frames this whole path exists to keep
    straight. The staleness notice already covers the reader here.
    """
    assert _decide(run_date=date(2026, 8, 16)) == NOT_TONIGHT


def test_a_ranking_from_a_later_night_is_not_rescored():
    """`--run-date` used deliberately, or a clock that disagrees."""
    assert _decide(run_date=date(2026, 8, 18)) == NOT_TONIGHT


def test_the_night_check_beats_the_forecast_check():
    assert (
        _decide(run_date=date(2026, 8, 16), forecast_issued_at=NOW - timedelta(days=3))
        == NOT_TONIGHT
    )


def test_a_forecast_issued_in_the_future_is_not_stale():
    """Clock skew must not trigger a write."""
    assert _decide(forecast_issued_at=NOW + timedelta(minutes=10)) == FORECAST_FRESH


class TestBatchIsRunning:
    """An unfinished row is not proof of a live process.

    Only `finish` closes it, so a killed batch leaves `completed_at` NULL
    forever. Read as "a batch is running", that would disable the rescore
    permanently and silently, with the remedy being a database edit nobody would
    know to make.
    """

    def test_no_open_run_is_not_running(self):
        assert batch_is_running(None, NOW) is False

    def test_a_run_that_began_an_hour_ago_is_running(self):
        assert batch_is_running(NOW - timedelta(hours=1), NOW) is True

    def test_a_run_that_began_within_a_normal_batch_is_running(self):
        """A batch takes about eight hours; seven in is still in flight."""
        assert batch_is_running(NOW - timedelta(hours=7), NOW) is True

    def test_a_run_open_since_yesterday_is_a_crash(self):
        assert batch_is_running(NOW - timedelta(hours=30), NOW) is False
