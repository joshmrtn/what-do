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

from src.models.preference_revision import PreferenceLine, PreferenceRevision
from src.storage.sqlite.connection import init_db
from src.storage.memory.preference_revisions import InMemoryPreferenceRevisionRepository
from src.storage.memory.runs import InMemoryRunRepository
from src.storage.sqlite.preference_revisions import SqlitePreferenceRevisionRepository
from src.storage.sqlite.runs import SqliteRunRepository

_START = datetime(2026, 8, 11, 2, 0, tzinfo=timezone.utc)


@pytest.fixture(params=["sqlite", "memory"])
def backend(request, tmp_path):
    """Both repositories over one store, so their foreign key is real.

    `run_history.preference_revision_id` references `preference_revisions(id)`,
    and SQLite enforces it — so a run's revision has to be one the revision
    repository actually minted, not a string the test made up.
    """
    if request.param == "sqlite":
        path = tmp_path / "runs.db"
        init_db(path)
        return SqliteRunRepository(path), SqlitePreferenceRevisionRepository(path)
    return InMemoryRunRepository(), InMemoryPreferenceRevisionRepository()


@pytest.fixture
def repo(backend):
    """One repository per implementation, so every test below runs twice."""
    return backend[0]


@pytest.fixture
def revisions(backend):
    """The revision repository over the same store as `repo`."""
    return backend[1]


def _a_revision(revisions) -> str:
    """Record a revision and return an id a run may reference."""
    return revisions.record(
        PreferenceRevision(
            captured_at=_START,
            content_hash="content-hash-for-run-provenance",
            lines=[
                PreferenceLine(
                    file_name="likes.txt",
                    position=0,
                    domain="general",
                    preference_type="like",
                    line_text="live music",
                    line_hash="line-hash",
                )
            ],
        )
    )


class TestStart:
    def test_starting_a_run_returns_an_id_that_identifies_it(self, repo):
        run_id = repo.start(_START)

        assert repo.open_run().run_id == run_id

    def test_two_runs_never_share_an_id(self, repo):
        assert repo.start(_START) != repo.start(_START)

    def test_a_run_records_which_preferences_it_scored_against(self, repo, revisions):
        """The other half of `scoring_config`.

        The constants say how the arithmetic ran; this says what it ran
        against. Both files are gitignored and edited freely, so a score whose
        preferences have since changed is otherwise unattributable.
        """
        revision_id = _a_revision(revisions)
        repo.start(_START, preference_revision_id=revision_id)

        assert repo.open_run().preference_revision_id == revision_id

    def test_every_provenance_argument_the_protocol_allows_is_accepted(
        self, repo, revisions
    ):
        """Both configs and the revision, together, in one call.

        The in-memory repository did not accept `dedup_config` at all while the
        protocol declared it and the batch passed it — a drift no test saw
        because no test had ever passed one to this fake.
        """
        run_id = repo.start(
            _START,
            scoring_config='{"gate_midpoint": 0.6}',
            dedup_config='{"decision_floor": 0.7}',
            preference_revision_id=_a_revision(revisions),
        )

        assert run_id


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


class TestScoringProvenance:
    """Which constants produced a score, recorded with the run that used them.

    `config/config.yaml` is gitignored, so the scoring constants live in neither
    version control nor the database. The moment one is tuned — and the tag
    confidence curve is explicitly expected to be re-fitted as nights
    accumulate — every past score becomes unexplainable: the number is stored,
    the arithmetic that produced it is not.
    """

    def test_the_scoring_config_is_stored_with_the_run(self, repo):
        run_id = repo.start(_START, scoring_config='{"tag_confidence_cap": 5.0}')

        assert repo.get(run_id).scoring_config == '{"tag_confidence_cap": 5.0}'

    def test_a_run_recorded_without_one_reads_back_as_none(self, repo):
        """A dry run or an older row simply has nothing to say."""
        run_id = repo.start(_START)

        assert repo.get(run_id).scoring_config is None


class TestTheLatestRun:
    """What `what-do --status` reports when nothing is running.

    Deliberately not `open_run`'s counterpart. That one hunts for a crash and
    must therefore ignore the successful run that came after it; this one
    answers *when did this system last do anything, and how did it go* — the
    newest row, finished or not.
    """

    def test_nothing_has_ever_run(self, repo):
        assert repo.latest() is None

    def test_the_newest_row_wins(self, repo, revisions):
        revision = _a_revision(revisions)
        older = repo.start(_START, preference_revision_id=revision)
        newer = repo.start(_START + timedelta(days=1), preference_revision_id=revision)
        repo.finish(older, outcome="success", completed_at=_START + timedelta(hours=8))

        assert repo.latest().run_id == newer

    def test_it_carries_the_outcome(self, repo, revisions):
        run_id = repo.start(_START, preference_revision_id=_a_revision(revisions))
        repo.finish(
            run_id, outcome="partial", completed_at=_START + timedelta(hours=8),
            errors=["extraction failed: no reply"],
        )

        latest = repo.latest()
        assert latest.outcome == "partial"
        assert latest.completed_at == _START + timedelta(hours=8)
        assert latest.errors == ["extraction failed: no reply"]

    def test_an_unfinished_run_is_still_the_latest(self, repo, revisions):
        """A run that is open is the last thing this system did, and reporting
        the one before it would describe a night that is over as current."""
        run_id = repo.start(_START, preference_revision_id=_a_revision(revisions))

        assert repo.latest().run_id == run_id
        assert repo.latest().completed_at is None


class TestReportingACrash:
    """A crash is announced once, and then it is old news.

    `--status` footnotes an unfinished run once a later run exists, so a death
    nobody examined cannot be lost. Left to itself that footnote never leaves:
    the run of 2026-08-12 was still on screen a week later, by which point every
    event it missed had been re-ingested or had simply happened.

    Stamping it is what lets the footnote clear without inventing an
    acknowledgement command. The stamp lives on the run because that is what it
    is a fact about.
    """

    def test_a_run_has_not_been_reported_when_it_starts(self, repo, revisions):
        run_id = repo.start(_START, preference_revision_id=_a_revision(revisions))

        assert repo.get(run_id).crash_reported_at is None

    def test_reporting_a_crash_records_when(self, repo, revisions):
        run_id = repo.start(_START, preference_revision_id=_a_revision(revisions))
        seen_at = _START + timedelta(days=7, hours=11)

        repo.mark_crash_reported(run_id, seen_at)

        assert repo.get(run_id).crash_reported_at == seen_at

    def test_the_first_report_is_the_one_kept(self, repo, revisions):
        """The question is *has this been told to anyone*, so a second telling
        must not overwrite when the first happened."""
        run_id = repo.start(_START, preference_revision_id=_a_revision(revisions))
        first = _START + timedelta(days=1)

        repo.mark_crash_reported(run_id, first)
        repo.mark_crash_reported(run_id, _START + timedelta(days=30))

        assert repo.get(run_id).crash_reported_at == first

    def test_an_unknown_run_id_is_ignored_rather_than_raising(self, repo):
        """Matching `finish`: reporting must never be what kills the reporter."""
        repo.mark_crash_reported("no-such-run", _START)

    def test_the_stamp_survives_finishing_the_run(self, repo, revisions):
        """A run reported as dead and later completed by hand keeps both facts."""
        run_id = repo.start(_START, preference_revision_id=_a_revision(revisions))
        seen_at = _START + timedelta(days=2)
        repo.mark_crash_reported(run_id, seen_at)

        repo.finish(run_id, outcome="failed", completed_at=_START + timedelta(days=3))

        record = repo.get(run_id)
        assert record.crash_reported_at == seen_at
        assert record.outcome == "failed"
