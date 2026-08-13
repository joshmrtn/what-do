from __future__ import annotations

import io
import json
from datetime import date, datetime, timezone
from pathlib import Path

import pytest

from src.ingestion.ingestion_service import RawCandidateRecord
from src.models.event_candidate import EventCandidate
from src.scheduler import BatchResult, run
from src.storage.schema_check import Finding, check_database
from src.storage.sqlite.connection import init_db

NOW = datetime(2026, 6, 15, 2, 0, 0, tzinfo=timezone.utc)


def _raw_record() -> RawCandidateRecord:
    return RawCandidateRecord(
        candidate=EventCandidate(
            id="c1", source="moon", source_type="moon", discovered_at=NOW
        ),
        source="moon",
        verdict="discarded",
        reason="out of window: start_time=None",
    )


class _Recorder:
    """Stands in for run_batch, capturing the kwargs the entry point built."""

    def __init__(self, result: BatchResult | None = None, raw: list | None = None) -> None:
        self.kwargs: dict = {}
        self.result = result or BatchResult(outcome="success")
        self.raw = raw

    def __call__(self, **kwargs):
        self.kwargs = kwargs
        # run_batch hands the dump its records mid-run, so the fake must too —
        # calling it afterwards would write to a stream nobody is reading.
        if self.raw is not None and kwargs.get("raw_dump_fn") is not None:
            kwargs["raw_dump_fn"](self.raw)
        return self.result


_STAGES = (
    "ingestion_service",
    "normalization_service",
    "enrichment_service",
    "extraction_stage",
    "embedding_stage",
    "semantic_deduplicator",
    "similarity_stage",
    "ranking_engine",
)


_REPOSITORIES = (
    "candidate_repository",
    "event_repository",
    "run_repository",
    "score_repository",
    "ranking_repository",
)


class _FakeDeps:
    """Stands in for the composition root, so no real provider is constructed."""

    def __init__(self, **built_with) -> None:
        self.built_with = built_with
        self.skipped_sources = ["apify"]
        for stage in _STAGES:
            setattr(self, stage, object())
        # Persistence now comes from the composition root too, so the
        # stand-in has to supply it like everything else.
        for repo in _REPOSITORIES:
            setattr(self, repo, object())


def _fake_build(**kwargs):
    return _FakeDeps(**kwargs)


@pytest.fixture
def invoke(tmp_path):
    """Drive run() with every real construction seam replaced."""

    def _invoke(argv, result=None, build=None, raw=None, check_schema=None):
        batch = _Recorder(result, raw=raw)
        config_calls: list = []

        def _load_config(config_path=None, env_path=None):
            config_calls.append(config_path)
            return object()

        stdout = io.StringIO()
        code = run(
            [*argv, "--db", str(tmp_path / "batch.db")] if "--db" not in argv else argv,
            get_now=lambda: NOW,
            stdout=stdout,
            load_config_fn=_load_config,
            build_dependencies_fn=build or _fake_build,
            run_batch_fn=batch,
            init_db_fn=lambda path: None,
            check_schema_fn=check_schema or (lambda path: []),
        )
        return code, batch, stdout.getvalue(), config_calls

    return _invoke


class _FakeTranscriptFactory:
    """Records the path chosen without opening a file."""

    def __init__(self) -> None:
        self.paths: list = []
        self.closed = False

    def __call__(self, path, get_now=None):
        self.paths.append(path)
        return self

    def close(self) -> None:
        self.closed = True


def _invoke_with_transcript(argv, tmp_path, factory):
    """Drive run() capturing what the composition root was handed."""
    built: dict = {}

    def _build(**kwargs):
        built.update(kwargs)
        return _FakeDeps(**kwargs)

    run(
        [*argv, "--db", str(tmp_path / "batch.db")],
        get_now=lambda: NOW,
        stdout=io.StringIO(),
        load_config_fn=lambda config_path=None, env_path=None: object(),
        build_dependencies_fn=_build,
        run_batch_fn=_Recorder(),
        init_db_fn=lambda path: None,
        # Stubbed alongside `init_db_fn`: this helper never creates the database
        # it names, and the real check opens it read-only.
        check_schema_fn=lambda path: [],
        transcript_factory=factory,
    )
    return built


def test_no_transcript_is_created_by_default(tmp_path):
    factory = _FakeTranscriptFactory()

    built = _invoke_with_transcript([], tmp_path, factory)

    assert factory.paths == []
    assert built["llm_transcript"] is None


def test_the_transcript_flag_names_a_file_after_the_run(tmp_path):
    factory = _FakeTranscriptFactory()

    built = _invoke_with_transcript(["--llm-transcript"], tmp_path, factory)

    assert built["llm_transcript"] is factory
    assert len(factory.paths) == 1
    assert factory.paths[0].name == "llm-20260615-020000.jsonl"
    assert factory.paths[0].parent.name == "logs"


def test_the_transcript_flag_accepts_an_explicit_path(tmp_path):
    factory = _FakeTranscriptFactory()
    target = tmp_path / "somewhere" / "calls.jsonl"

    _invoke_with_transcript(["--llm-transcript", str(target)], tmp_path, factory)

    assert factory.paths == [target]


def test_the_transcript_is_closed_when_the_run_ends(tmp_path):
    factory = _FakeTranscriptFactory()

    _invoke_with_transcript(["--llm-transcript"], tmp_path, factory)

    assert factory.closed is True


def test_the_run_date_defaults_to_today(invoke):
    _, batch, _, _ = invoke([])

    assert batch.kwargs["run_date"] == date(2026, 6, 15)


def test_an_explicit_run_date_is_used(invoke):
    _, batch, _, _ = invoke(["--run-date", "2026-07-04"])

    assert batch.kwargs["run_date"] == date(2026, 7, 4)


def test_a_malformed_run_date_fails_rather_than_guessing(invoke):
    code, batch, _, _ = invoke(["--run-date", "the fourth"])

    assert code != 0
    assert batch.kwargs == {}


def test_dry_run_is_passed_through(invoke):
    _, batch, _, _ = invoke(["--dry-run"])

    assert batch.kwargs["dry_run"] is True


def test_skip_ingest_is_passed_through(invoke):
    _, batch, _, _ = invoke(["--skip-ingest"])

    assert batch.kwargs["skip_ingest"] is True


def test_neither_flag_is_set_by_default(invoke):
    _, batch, _, _ = invoke([])

    assert batch.kwargs["dry_run"] is False
    assert batch.kwargs["skip_ingest"] is False


def test_the_config_path_reaches_the_loader(invoke):
    _, _, _, config_calls = invoke(["--config", "/etc/what-do.yaml"])

    assert config_calls == [Path("/etc/what-do.yaml")]


def test_the_db_flag_chooses_the_database(invoke, tmp_path):
    target = tmp_path / "elsewhere.db"

    _, batch, _, _ = invoke(["--db", str(target)])

    assert batch.kwargs["db_path"] == target


def test_skipped_sources_are_reported_not_inferred(invoke):
    """The composition root knows what it could not build; run_batch is told."""
    _, batch, _, _ = invoke([])

    assert batch.kwargs["skipped_sources"] == ["apify"]


def test_a_successful_run_exits_zero(invoke):
    code, _, _, _ = invoke([])

    assert code == 0


def test_a_partial_run_still_exits_zero(invoke):
    """A stage failed, but recommendations were produced; that is not a failure."""
    code, _, _, _ = invoke([], result=BatchResult(outcome="partial", errors=["x failed"]))

    assert code == 0


def test_a_failed_run_exits_non_zero(invoke):
    """Cron needs to see that the batch stopped before ranking."""
    code, _, _, _ = invoke([], result=BatchResult(outcome="failed"))

    assert code != 0


def test_the_summary_reports_the_outcome_and_counts(invoke):
    result = BatchResult(
        outcome="partial",
        stage_counts={"ingested": 12, "ranked": 7},
        errors=["enrichment failed: boom"],
        skipped_sources=["apify"],
    )

    _, _, output, _ = invoke([], result=result)

    assert "partial" in output
    assert "7" in output
    assert "enrichment failed: boom" in output
    assert "apify" in output


# ----------------------------------------------------------------------
# Ingest-only and the raw dump
# ----------------------------------------------------------------------


def test_ingest_only_is_passed_through(invoke):
    _, batch, _, _ = invoke(["--ingest-only"])

    assert batch.kwargs["ingest_only"] is True


def test_ingest_only_defaults_off(invoke):
    _, batch, _, _ = invoke([])

    assert batch.kwargs["ingest_only"] is False


def test_no_raw_dump_is_requested_by_default(invoke):
    """Collecting the raw fetch costs memory, so nothing asks for it unbidden."""
    _, batch, _, _ = invoke([])

    assert batch.kwargs["raw_dump_fn"] is None


def test_raw_without_a_path_dumps_to_stdout(invoke):
    _, _, out, _ = invoke(["--raw"], raw=[_raw_record()])

    assert '"source": "moon"' in out
    assert '"verdict": "discarded"' in out


def test_raw_with_a_path_writes_a_file(invoke, tmp_path):
    target = tmp_path / "raw.jsonl"

    _, _, out, _ = invoke(["--raw", str(target)], raw=[_raw_record(), _raw_record()])

    lines = target.read_text().strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["candidate"]["id"] == "c1"
    assert out.strip().startswith("outcome:")


def test_raw_renders_times_as_text(invoke, tmp_path):
    """A datetime is not JSON, and a dump that raises explains nothing."""
    target = tmp_path / "raw.jsonl"

    invoke(["--raw", str(target)], raw=[_raw_record()])

    record = json.loads(target.read_text().strip())
    assert record["candidate"]["discovered_at"] == NOW.isoformat()


def test_the_default_clock_is_timezone_aware():
    """A naive clock meets an aware start_time and the whole fetch raises.

    Sources that state their own offset — Do617, every ICS feed — are compared
    against this clock during ingestion. `datetime.now` returns a naive time,
    which killed the first real run at the ingestion stage.
    """
    from src.scheduler import _default_now

    assert _default_now().tzinfo is not None


class TestTheSchemaGate:
    """`_SCHEMA` is all `CREATE TABLE IF NOT EXISTS`, so a new column serves
    every fresh database and leaves the live file untouched until a hand
    migration runs. Every test passes either way. This is the one check that can
    see the difference, so it runs before anything else does.
    """

    def test_a_drifted_schema_stops_the_batch(self, invoke):
        """Abort rather than report. A drift that reaches a stage kills the run
        minutes later anyway, having burnt model time first — aborting only
        moves the death to the cheap end."""
        drift = [Finding("events", "events.extraction_degradation is missing")]
        builds: list = []

        code, batch, out, _ = invoke(
            [],
            build=lambda **kw: builds.append(kw),
            check_schema=lambda path: drift,
        )

        assert code == 1
        assert builds == [], "dependencies were built despite the drift"
        assert batch.kwargs == {}, "the batch ran despite the drift"
        assert "extraction_degradation" in out

    def test_a_clean_schema_lets_the_batch_proceed(self, invoke):
        code, batch, _, _ = invoke([], check_schema=lambda path: [])

        assert code == 0
        assert batch.kwargs, "the batch never ran"

    def test_the_gate_applies_to_a_dry_run_too(self, invoke):
        """The tempting exception, and the wrong one: a dry run exists to say
        the pipeline is sound, so one that passes against a schema the real run
        would die on gives exactly the false confidence it exists to prevent."""
        code, batch, _, _ = invoke(
            ["--dry-run"], check_schema=lambda path: [Finding("events", "drift")]
        )

        assert code == 1
        assert batch.kwargs == {}

    def test_the_gate_applies_to_ingest_only_too(self, invoke):
        """`--ingest-only` writes candidates, so it was never in the safe
        category to begin with."""
        code, batch, _, _ = invoke(
            ["--ingest-only"], check_schema=lambda path: [Finding("events", "drift")]
        )

        assert code == 1
        assert batch.kwargs == {}

    def test_the_real_check_is_the_default(self, tmp_path):
        """A seam only tests reach is a seam production never exercises — the
        naive-clock failure exactly. So the default is asserted against a real
        database on disk."""
        db = tmp_path / "clean.db"
        init_db(db)

        assert check_database(db) == []
