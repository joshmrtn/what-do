"""What `what-do --status` says, in each of the four states it can find."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from src.models.run import RunRecord
from src.observability.heartbeat import Heartbeat, HeartbeatFile, Item
from src.observability.reporter import FINISHED, Progress
from src.presentation.status import StatusInputs, probe_status, render_status

_START = datetime(2026, 8, 18, 6, 0, tzinfo=timezone.utc)
_STALL_AFTER = timedelta(minutes=15)


def _at(minutes: float) -> datetime:
    return _START + timedelta(minutes=minutes)


def _beat(**kwargs) -> Heartbeat:
    defaults = dict(
        run_id="run-1",
        stage="extraction",
        done=253,
        total=745,
        started_at=_START,
        updated_at=_at(241),
        in_flight=Item("evt-9", "Salem Jazz & Soul Festival", _at(241)),
        last_completed=Item("evt-8", "Trivia at the Rhumb Line", _at(240)),
        deadline=_at(480),
    )
    defaults.update(kwargs)
    return Heartbeat(**defaults)


def _run(**kwargs) -> RunRecord:
    defaults = dict(run_id="run-1", started_at=_START)
    defaults.update(kwargs)
    return RunRecord(**defaults)


def _render(**kwargs) -> str:
    defaults = dict(
        lock_held=True,
        heartbeat=_beat(),
        heartbeat_run=_run(),
        open_run=_run(),
        latest_run=_run(),
        now=_at(242),
        stall_after=_STALL_AFTER,
        zone=timezone.utc,
    )
    defaults.update(kwargs)
    return render_status(**defaults)


class TestRunning:
    def test_it_says_so_and_says_how_far(self):
        out = _render()

        assert out.startswith("running")
        assert "extraction 253/745 (34%)" in out

    def test_it_names_what_the_model_has_right_now(self):
        """The question a log cannot answer: not what has been done, but what
        is happening at the moment of asking."""
        assert "Salem Jazz & Soul Festival" in _render()

    def test_it_reports_the_rate_and_the_budget(self):
        out = _render()

        assert "57s each" in out, out
        assert "3h58m of budget left" in out, out

    def test_an_unbounded_run_projects_on_the_rate_alone(self):
        """With no budget the queue is the only thing that binds, and how long
        it will take is the whole question."""
        out = _render(heartbeat=_beat(deadline=None))

        assert "budget" not in out
        assert "7h50m" in out, out

    def test_a_bounded_run_says_which_of_the_two_binds(self):
        """Under a slow provider the budget stops the run with the queue half
        done; under a fast one the queue finishes first. One line, both
        readings — and it is the same arithmetic either way."""
        out = _render()

        assert "budget binds first" in out, out
        assert "deferred to tomorrow" in out, out

    def test_a_run_that_will_finish_its_queue_says_that_instead(self):
        out = _render(heartbeat=_beat(done=700, updated_at=_at(241)))

        assert "budget binds first" not in out
        assert "on course to finish" in out, out

    def test_nothing_finished_yet_does_not_divide_by_zero(self):
        out = _render(heartbeat=_beat(done=0, last_completed=None))

        assert "0/745" in out
        assert "each" not in out, "there is no rate to report yet"

    def test_a_stage_with_no_budget_of_its_own_still_reports(self):
        out = _render(heartbeat=_beat(stage="embedding", deadline=None, total=40, done=12))

        assert "embedding 12/40 (30%)" in out


class TestStalled:
    """The state nothing could see before: the lock is held, the log is silent,
    and a healthy slow run looks exactly like a hung one."""

    def test_an_item_in_flight_too_long_is_stuck_not_slow(self):
        out = _render(
            heartbeat=_beat(in_flight=Item("evt-9", "Salem Jazz", _at(200))),
            now=_at(242),
        )

        assert out.startswith("stalled")
        assert "42m" in out

    def test_the_threshold_is_the_configured_one(self):
        beat = _beat(in_flight=Item("evt-9", "Salem Jazz", _at(220)))

        assert _render(heartbeat=beat, now=_at(242), stall_after=timedelta(minutes=15)
                       ).startswith("stalled")
        assert _render(heartbeat=beat, now=_at(242), stall_after=timedelta(hours=2)
                       ).startswith("running")

    def test_a_run_between_items_is_not_stalled(self):
        """Nothing in flight means the stage is choosing its next item, which
        takes no time at all — there is nothing to be stuck on."""
        out = _render(heartbeat=_beat(in_flight=None), now=_at(400))

        assert out.startswith("running")


class TestDied:
    def test_a_heartbeat_without_a_process_is_a_death(self):
        out = _render(lock_held=False)

        assert out.startswith("died")

    def test_it_names_the_event_the_batch_died_on(self):
        """The whole reason each item is reported twice. The transcript records
        calls only once they return, so an event the process was killed inside
        appears in no other artefact."""
        out = _render(lock_held=False)

        assert "Salem Jazz & Soul Festival" in out
        assert "evt-9" in out, "an id, because a label cannot be looked up"

    def test_a_run_that_died_before_reporting_anything_is_still_a_death(self):
        """Ingestion and normalization come before extraction, so a batch can
        die having reported nothing at all. The open run row is the evidence."""
        open_run = _run(run_id="run-1")
        out = _render(
            lock_held=False, heartbeat=None, heartbeat_run=None,
            open_run=open_run, latest_run=open_run,
        )

        assert out.startswith("died")
        assert "before it reported any progress" in out

    def test_a_crash_with_successful_runs_after_it_is_history_not_news(self):
        """Found by running this against the real database: `open_run` returns
        the newest *unfinished* row, and the live one is 2026-08-12 — the night
        the batch died, six successful runs ago. That is right for `open_run`,
        whose question is *has a crash gone unexamined*, and wrong for
        `--status`, whose question is *what is happening now*. An alarm that
        can never clear is one nobody reads."""
        crash = _run(run_id="run-old", started_at=_START - timedelta(days=6))
        latest = _run(
            run_id="run-new", completed_at=_at(480), outcome="success"
        )
        out = _render(
            lock_held=False, heartbeat=None, heartbeat_run=None,
            open_run=crash, latest_run=latest, now=_at(600),
        )

        assert out.startswith("idle")

    def test_an_older_crash_is_still_mentioned(self):
        """Not hidden, either. `open_run` exists so an unexamined crash cannot
        be lost, and the idle line is where it belongs — as a footnote rather
        than as the headline."""
        crash = _run(run_id="run-old", started_at=_START - timedelta(days=6))
        out = _render(
            lock_held=False, heartbeat=None, heartbeat_run=None,
            open_run=crash, latest_run=_run(run_id="run-new", completed_at=_at(480),
                                            outcome="success"),
            now=_at(600),
        )

        assert "2026-08-12" in out
        assert "never finished" in out

    def test_a_leftover_from_a_finished_run_is_not_a_death(self):
        """A file whose run has a `completed_at` is the remains of a batch that
        finished and could not clean up. Reading that as a death would cry wolf
        every morning."""
        finished = _run(completed_at=_at(480), outcome="success")
        out = _render(
            lock_held=False, heartbeat_run=finished, open_run=None, latest_run=finished
        )

        assert out.startswith("idle")


class TestTheZone:
    """Timestamps are stored in UTC and the batch runs at 02:00 local. A status
    line reading `06:00` for it is the kind of wrong that looks right."""

    def test_clock_times_are_shown_in_the_view_s_zone(self):
        finished = _run(completed_at=_at(480), outcome="success")
        out = _render(
            lock_held=False, heartbeat=None, heartbeat_run=None,
            open_run=None, latest_run=finished, now=_at(600),
            zone=timezone(timedelta(hours=-4)),
        )

        assert "2026-08-18 02:00 → 10:00" in out, out

    def test_the_event_a_batch_died_on_is_stamped_locally_too(self):
        out = _render(lock_held=False, zone=timezone(timedelta(hours=-4)))

        assert "in flight since 06:01" in out, out


class TestIdle:
    def test_it_reports_the_last_run_and_its_age(self):
        finished = _run(completed_at=_at(480), outcome="success")
        out = _render(
            lock_held=False, heartbeat=None, heartbeat_run=None,
            open_run=None, latest_run=finished, now=_at(600),
        )

        assert out.startswith("idle")
        assert "success" in out
        assert "2h00m ago" in out, out

    def test_a_failed_last_run_says_what_went_wrong(self):
        finished = _run(
            completed_at=_at(480), outcome="partial",
            errors=["extraction failed: no reply from model"],
        )
        out = _render(
            lock_held=False, heartbeat=None, heartbeat_run=None,
            open_run=None, latest_run=finished, now=_at(600),
        )

        assert "partial" in out
        assert "extraction failed: no reply from model" in out

    def test_a_system_that_has_never_run(self):
        out = _render(
            lock_held=False, heartbeat=None, heartbeat_run=None,
            open_run=None, latest_run=None,
        )

        assert "no batch has ever run" in out


class TestWhatCannotBeRead:
    def test_a_lock_held_with_no_heartbeat_is_running_without_detail(self):
        """True for the first minutes of every run: ingestion, normalization
        and enrichment all come before the first extraction."""
        out = _render(heartbeat=None, heartbeat_run=None)

        assert out.startswith("running")
        assert "no progress reported yet" in out

    def test_an_unreadable_heartbeat_never_raises(self):
        """`read_heartbeat` returns None for a truncated file, and a `--status`
        that failed on one would be worse than one that admits it cannot tell."""
        out = _render(heartbeat=None, heartbeat_run=None, lock_held=True)

        assert out


class TestTheProbe:
    """Reading the three signals, including from a machine that has none."""

    def test_a_database_that_does_not_exist_yet_still_answers(self, tmp_path):
        """`--status` is exactly what someone runs before the first batch has
        finished. Without a run history the lock and the heartbeat still answer
        most of the question."""
        inputs = probe_status(
            None,
            lock_path=tmp_path / "absent.lock",
            heartbeat_path=tmp_path / "absent.json",
        )

        assert inputs == StatusInputs(
            lock_held=False, heartbeat=None,
            heartbeat_run=None, open_run=None, latest_run=None,
        )

    def test_it_asks_the_history_about_the_run_the_heartbeat_names(self, tmp_path):
        """The leftover check. Without it, a heartbeat that outlived a clean
        exit would be reported as a death every morning."""
        asked = []

        class _Runs:
            def get(self, run_id):
                asked.append(run_id)
                return _run(run_id=run_id, completed_at=_at(480))

            def open_run(self):
                return None

            def latest(self):
                return None

        path = tmp_path / "progress.json"
        HeartbeatFile(path, run_id="run-77")(
            Progress(
                stage="extraction", done=1, total=2, item_id="e", label="E",
                phase=FINISHED, now=_START,
            )
        )

        inputs = probe_status(
            _Runs(), lock_path=tmp_path / "absent.lock", heartbeat_path=path
        )

        assert asked == ["run-77"]
        assert inputs.heartbeat_run.completed_at == _at(480)
