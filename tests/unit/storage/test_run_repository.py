"""Contract every RunRepository implementation must satisfy.

Run against both the SQLite repository and the in-memory one, for the reasons
given in `test_event_repository.py`: the in-memory implementation is the single
official fake, and a hand-written one drifts from the contract silently.

`run_history` is the only durable record of what a 2am run did, and its whole
point is surviving the run's own death — so the behaviour that matters most here
is what a *half-written* row means.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from src.storage.db import init_db
from src.storage.memory.runs import InMemoryRunRepository
from src.storage.sqlite.runs import SqliteRunRepository

_START = datetime(2026, 8, 11, 2, 0, tzinfo=timezone.utc)


@pytest.fixture(params=["sqlite", "memory"])
def repo(request, tmp_path):
    """One repository per implementation, so every test below runs twice."""
    if request.param == "sqlite":
        path = tmp_path / "runs.db"
        init_db(path)
        return SqliteRunRepository(path)
    return InMemoryRunRepository()


class TestStart:
    def test_starting_a_run_returns_an_id_that_identifies_it(self, repo):
        run_id = repo.start(_START)

        assert repo.open_run().run_id == run_id

    def test_two_runs_never_share_an_id(self, repo):
        assert repo.start(_START) != repo.start(_START)


class TestOpenRun:
    """A row with a started_at and no completed_at is a crash.

    No end-of-run write can record a process that was killed, so this query is
    the only way the system can ever know a batch died mid-flight.
    """

    def test_there_is_no_open_run_before_anything_starts(self, repo):
        assert repo.open_run() is None

    def test_a_started_run_is_open_and_carries_when_it_began(self, repo):
        repo.start(_START)

        open_run = repo.open_run()

        assert open_run is not None
        assert open_run.started_at == _START

    def test_a_finished_run_is_no_longer_open(self, repo):
        run_id = repo.start(_START)

        repo.finish(run_id, outcome="success", completed_at=_START + timedelta(hours=1))

        assert repo.open_run() is None

    def test_the_most_recent_of_several_crashes_is_the_one_reported(self, repo):
        repo.start(_START)
        repo.start(_START + timedelta(days=1))

        assert repo.open_run().started_at == _START + timedelta(days=1)

    def test_a_finished_run_does_not_mask_an_earlier_crash(self, repo):
        # The crash is older than the run that succeeded after it. Reporting
        # only the latest row would say all is well while evidence of a dead
        # batch sits unexamined.
        repo.start(_START)
        later = repo.start(_START + timedelta(days=1))

        repo.finish(later, outcome="success", completed_at=_START + timedelta(days=1, hours=1))

        assert repo.open_run().started_at == _START


class TestFinish:
    def test_an_unknown_run_id_is_ignored_rather_than_raising(self, repo):
        # The batch must never die trying to record that it died.
        repo.finish("no-such-run", outcome="failed", completed_at=_START)

    def test_duration_is_derived_from_the_stored_start_not_a_passed_value(self, repo):
        # A resumed process still records real elapsed time.
        run_id = repo.start(_START)

        repo.finish(run_id, outcome="success", completed_at=_START + timedelta(minutes=90))

        assert repo.get(run_id).duration_ms == 90 * 60 * 1000

    def test_counts_errors_and_skips_survive_a_round_trip(self, repo):
        run_id = repo.start(_START)

        repo.finish(
            run_id,
            outcome="partial",
            completed_at=_START + timedelta(hours=1),
            stage_counts={"events": 1180, "ranked": 860},
            errors=["enrichment blew up"],
            skipped_sources=["amc"],
        )

        record = repo.get(run_id)
        assert record.outcome == "partial"
        assert record.stage_counts == {"events": 1180, "ranked": 860}
        assert record.errors == ["enrichment blew up"]
        assert record.skipped_sources == ["amc"]

    def test_a_skip_is_recorded_apart_from_an_error(self, repo):
        # A skip is a legitimate deployment state — no key for that source.
        # Folding it in with failures loses the distinction the credential
        # policy rests on.
        run_id = repo.start(_START)

        repo.finish(
            run_id,
            outcome="success",
            completed_at=_START + timedelta(hours=1),
            skipped_sources=["amc"],
        )

        record = repo.get(run_id)
        assert record.skipped_sources == ["amc"]
        assert record.errors == []

    def test_omitted_counts_and_errors_read_back_empty_not_absent(self, repo):
        run_id = repo.start(_START)

        repo.finish(run_id, outcome="success", completed_at=_START + timedelta(hours=1))

        record = repo.get(run_id)
        assert record.stage_counts == {}
        assert record.errors == []
        assert record.skipped_sources == []


class TestGet:
    def test_an_unknown_run_id_reads_back_as_nothing(self, repo):
        assert repo.get("no-such-run") is None

    def test_an_unfinished_run_reads_back_without_an_outcome(self, repo):
        run_id = repo.start(_START)

        record = repo.get(run_id)

        assert record.completed_at is None
        assert record.outcome is None
        assert record.duration_ms is None
