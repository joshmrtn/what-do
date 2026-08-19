"""How a per-item report becomes the few lines a person will actually read."""

from __future__ import annotations

import io
from datetime import datetime, timedelta, timezone

import pytest

from src.observability.reporter import (
    FINISHED,
    STARTED,
    Progress,
    ProgressLog,
    format_duration,
)
from src.utils.logging import get_logger


def _at(minutes: float) -> datetime:
    return datetime(2026, 8, 18, 6, 0, tzinfo=timezone.utc) + timedelta(minutes=minutes)


def _report(done, *, total=100, at=0.0, phase=FINISHED, deadline=None, stage="extraction"):
    return Progress(
        stage=stage,
        done=done,
        total=total,
        item_id=f"evt-{done}",
        label=f"Event {done}",
        phase=phase,
        now=_at(at),
        deadline=deadline,
    )


class _Log:
    """The real logger, writing where a test can read it."""

    def __init__(self):
        self.stream = io.StringIO()
        self.logger = get_logger("test_progress", stream=self.stream)

    @property
    def lines(self) -> list[str]:
        return [
            line for line in self.stream.getvalue().splitlines()
            if "extraction " in line or "embedding " in line
        ]


def _drive(policy, reports):
    for report in reports:
        policy(report)


class TestTheMilestone:
    def test_a_line_on_the_boundary_and_not_per_item(self):
        log = _Log()
        policy = ProgressLog(log.logger, milestone_fraction=0.25, heartbeat=timedelta(hours=99))

        _drive(policy, [_report(n, at=n) for n in range(1, 101)])

        assert len(log.lines) == 4, "quarters of the queue, not one a piece"

    def test_the_line_names_what_is_done_and_what_is_left(self):
        log = _Log()
        policy = ProgressLog(log.logger, milestone_fraction=0.5, heartbeat=timedelta(hours=99))

        _drive(policy, [_report(n, at=n) for n in range(1, 51)])

        assert "extraction 50/100 (50%)" in log.lines[0]

    def test_the_final_item_reports(self):
        """A run that finishes its queue says so; without it the last thing the
        log holds is 75% and a reader cannot tell finished from abandoned."""
        log = _Log()
        policy = ProgressLog(log.logger, milestone_fraction=0.25, heartbeat=timedelta(hours=99))

        _drive(policy, [_report(n, at=n) for n in range(1, 101)])

        assert "100/100 (100%)" in log.lines[-1]

    def test_a_started_report_never_logs(self):
        """Started is for the heartbeat file, which answers *what is happening
        now*. The log answers *what has been done*, and an item that has only
        begun has done nothing."""
        log = _Log()
        policy = ProgressLog(log.logger, milestone_fraction=0.25, heartbeat=timedelta(minutes=1))

        _drive(policy, [_report(0, at=n, phase=STARTED) for n in range(1, 30)])

        assert log.lines == []


class TestTheHeartbeat:
    def test_silence_alone_is_enough_to_speak(self):
        """The case the whole sprint exists for: 25% of an eight-hour queue is
        two hours away, and two hours of silence reads as a dead batch."""
        log = _Log()
        policy = ProgressLog(log.logger, milestone_fraction=0.25, heartbeat=timedelta(minutes=20))

        _drive(policy, [_report(n, total=1000, at=n * 10) for n in range(1, 7)])

        assert len(log.lines) == 2, "one every 20 minutes of the 50 it ran for"

    def test_the_first_line_comes_a_heartbeat_after_the_stage_starts(self):
        log = _Log()
        policy = ProgressLog(log.logger, milestone_fraction=0.25, heartbeat=timedelta(minutes=20))

        _drive(policy, [_report(n, total=1000, at=n) for n in range(1, 25)])

        assert len(log.lines) == 1
        assert "21/1000" in log.lines[0]

    def test_a_milestone_and_a_heartbeat_at_once_write_one_line(self):
        log = _Log()
        policy = ProgressLog(log.logger, milestone_fraction=0.25, heartbeat=timedelta(minutes=20))

        _drive(policy, [_report(n, at=n * 30) for n in range(1, 26)])

        assert len([line for line in log.lines if "25/100" in line]) == 1


class TestWhatTheLineSays:
    def test_elapsed_and_rate_come_from_the_reported_clock(self):
        """Never wall-clock arithmetic inside the sink. The stage's injected
        clock is the only one that knows what time it is for this run."""
        log = _Log()
        policy = ProgressLog(log.logger, milestone_fraction=1.0, heartbeat=timedelta(hours=99))

        # Bracketed the way the stage reports: an item starts, then finishes.
        reports = []
        for n in range(1, 5):
            reports.append(_report(n - 1, total=4, at=(n - 1) * 2, phase=STARTED))
            reports.append(_report(n, total=4, at=n * 2))
        _drive(policy, reports)

        line = log.lines[0]
        assert "8m elapsed" in line, line
        assert "120s each" in line, line

    def test_a_budget_deadline_becomes_time_remaining(self):
        log = _Log()
        policy = ProgressLog(log.logger, milestone_fraction=1.0, heartbeat=timedelta(hours=99))

        _drive(policy, [_report(1, total=1, at=10, deadline=_at(135))])

        assert "2h05m of budget left" in log.lines[0]

    def test_an_unbounded_stage_says_nothing_about_a_budget(self):
        """`extraction_budget_minutes` absent is a real configuration, and a
        line reading `0m of budget left` would be a lie about it."""
        log = _Log()
        policy = ProgressLog(log.logger, milestone_fraction=1.0, heartbeat=timedelta(hours=99))

        _drive(policy, [_report(1, total=1, at=10)])

        assert "budget" not in log.lines[0]

    def test_an_overrun_budget_does_not_report_negative_time(self):
        log = _Log()
        policy = ProgressLog(log.logger, milestone_fraction=1.0, heartbeat=timedelta(hours=99))

        _drive(policy, [_report(1, total=1, at=100, deadline=_at(40))])

        assert "budget spent" in log.lines[0]

    def test_the_stage_names_itself(self):
        log = _Log()
        policy = ProgressLog(log.logger, milestone_fraction=1.0, heartbeat=timedelta(hours=99))

        _drive(policy, [_report(1, total=1, at=1, stage="embedding")])

        assert log.lines[0].split()[0] == "embedding" or "embedding 1/1" in log.lines[0]


class TestASecondPass:
    def test_a_fresh_pass_starts_its_own_reckoning(self):
        """The same stage object runs once a batch, but a read-path rescore
        drives the embedding stage again in the same process. Carrying the
        first pass's start forward would report an elapsed time that includes
        the hours between them."""
        log = _Log()
        policy = ProgressLog(log.logger, milestone_fraction=1.0, heartbeat=timedelta(hours=99))

        _drive(policy, [
            _report(0, total=1, at=0, phase=STARTED), _report(1, total=1, at=0),
        ])
        _drive(policy, [
            _report(0, total=1, at=600, phase=STARTED), _report(1, total=1, at=600),
        ])

        assert "0s elapsed" in log.lines[1], log.lines


class TestFormatDuration:
    @pytest.mark.parametrize("seconds,expected", [
        (0, "0s"),
        (45, "45s"),
        (60, "1m"),
        (3599, "59m"),
        (3600, "1h00m"),
        (7500, "2h05m"),
        (86_400, "24h00m"),
    ])
    def test_it_reads_honestly_at_every_scale(self, seconds, expected):
        assert format_duration(timedelta(seconds=seconds)) == expected
