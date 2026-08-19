"""The file a running batch leaves behind, and what it says when it stops."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

from src.observability.heartbeat import HeartbeatFile, read_heartbeat
from src.observability.reporter import FINISHED, STARTED, Progress


def _at(minutes: float) -> datetime:
    return datetime(2026, 8, 18, 6, 0, tzinfo=timezone.utc) + timedelta(minutes=minutes)


def _report(done, *, phase=FINISHED, at=0.0, total=745, item="evt-1",
            label="Salem Jazz & Soul Festival", deadline=None):
    return Progress(
        stage="extraction", done=done, total=total, item_id=item, label=label,
        phase=phase, now=_at(at), deadline=deadline,
    )


def _file(tmp_path, run_id="run-1"):
    return HeartbeatFile(tmp_path / "progress.json", run_id=run_id)


class TestWhatItRecords:
    def test_a_started_item_is_in_flight(self, tmp_path):
        beat = _file(tmp_path)

        beat(_report(252, phase=STARTED, at=1))

        state = read_heartbeat(tmp_path / "progress.json")
        assert state.in_flight.item_id == "evt-1"
        assert state.in_flight.label == "Salem Jazz & Soul Festival"
        assert state.in_flight.since == _at(1)

    def test_a_finished_item_is_no_longer_in_flight(self, tmp_path):
        beat = _file(tmp_path)

        beat(_report(252, phase=STARTED, at=1))
        beat(_report(253, phase=FINISHED, at=3))

        state = read_heartbeat(tmp_path / "progress.json")
        assert state.in_flight is None
        assert state.last_completed.item_id == "evt-1"
        assert state.last_completed.since == _at(3)

    def test_done_counts_what_finished_not_what_started(self, tmp_path):
        """An item in the model's hands is not progress. Counting it would make
        a batch that hangs on its first event look one event further on than a
        batch that has done nothing."""
        beat = _file(tmp_path)

        beat(_report(252, phase=STARTED, at=1))

        assert read_heartbeat(tmp_path / "progress.json").done == 252

    def test_the_run_it_belongs_to_is_recorded(self, tmp_path):
        """What tells a leftover from a death. A file whose run has a
        `completed_at` is the remains of a batch that finished and could not
        clean up; one whose run is still open is a batch that died."""
        beat = _file(tmp_path, run_id="run-42")

        beat(_report(1, at=1))

        assert read_heartbeat(tmp_path / "progress.json").run_id == "run-42"

    def test_the_stage_and_its_queue_are_recorded(self, tmp_path):
        beat = _file(tmp_path)

        beat(_report(1, at=1, total=745))

        state = read_heartbeat(tmp_path / "progress.json")
        assert (state.stage, state.total) == ("extraction", 745)

    def test_the_start_of_the_pass_survives_later_reports(self, tmp_path):
        """So a reader can work out a rate without the log."""
        beat = _file(tmp_path)

        beat(_report(0, phase=STARTED, at=0))
        beat(_report(1, at=10))
        beat(_report(2, at=20))

        assert read_heartbeat(tmp_path / "progress.json").started_at == _at(0)

    def test_a_budget_deadline_is_recorded_when_there_is_one(self, tmp_path):
        beat = _file(tmp_path)

        beat(_report(1, at=1, deadline=_at(480)))

        assert read_heartbeat(tmp_path / "progress.json").deadline == _at(480)

    def test_an_unbounded_stage_records_no_deadline(self, tmp_path):
        beat = _file(tmp_path)

        beat(_report(1, at=1))

        assert read_heartbeat(tmp_path / "progress.json").deadline is None


class TestTheFileItself:
    def test_a_clean_finish_removes_it(self, tmp_path):
        """The whole three-state answer rests on this: a file left behind is
        the evidence of a process that could not clean up after itself."""
        beat = _file(tmp_path)
        beat(_report(1, at=1))

        beat.clear()

        assert read_heartbeat(tmp_path / "progress.json") is None

    def test_clearing_a_file_that_is_already_gone_is_not_an_error(self, tmp_path):
        """It runs in the batch's exit path, where raising would turn a
        successful run into a failed one over a file nobody needs."""
        _file(tmp_path).clear()

    def test_writing_leaves_no_temporary_behind(self, tmp_path):
        """Written to a temp path and renamed, so a reader never sees a torn
        document — `--status` reads this file while it is being written."""
        beat = _file(tmp_path)

        beat(_report(1, at=1))
        beat(_report(2, at=2))

        assert [p.name for p in tmp_path.iterdir()] == ["progress.json"]

    def test_an_absent_file_reads_as_nothing_running(self, tmp_path):
        assert read_heartbeat(tmp_path / "nothing.json") is None

    def test_a_truncated_file_reads_as_unreadable_rather_than_raising(self, tmp_path):
        """A `--status` that raises on a half-written file is worse than one
        that says it cannot tell."""
        path = tmp_path / "progress.json"
        path.write_text('{"run_id": "run-1", "done"')

        assert read_heartbeat(path) is None

    def test_a_document_missing_its_fields_reads_as_unreadable(self, tmp_path):
        path = tmp_path / "progress.json"
        path.write_text('{"hello": "world"}')

        assert read_heartbeat(path) is None

    def test_the_document_is_plain_json_a_person_can_read(self, tmp_path):
        """It lives in /tmp beside the lock and gets looked at by hand."""
        beat = _file(tmp_path)

        beat(_report(253, phase=STARTED, at=1, deadline=_at(480)))

        raw = json.loads((tmp_path / "progress.json").read_text())
        assert raw["stage"] == "extraction"
        assert raw["in_flight"]["label"] == "Salem Jazz & Soul Festival"

    def test_a_failure_to_write_never_stops_the_batch(self, tmp_path):
        """Reporting is a courtesy to whoever is watching. A full disk must not
        cost eight hours of model time."""
        beat = HeartbeatFile(tmp_path / "no-such-dir" / "progress.json", run_id="run-1")

        beat(_report(1, at=1))
