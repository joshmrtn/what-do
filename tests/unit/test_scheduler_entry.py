from __future__ import annotations

import io
from datetime import date, datetime, timezone
from pathlib import Path

import pytest

from src.scheduler import BatchResult, run

NOW = datetime(2026, 6, 15, 2, 0, 0, tzinfo=timezone.utc)


class _Recorder:
    """Stands in for run_batch, capturing the kwargs the entry point built."""

    def __init__(self, result: BatchResult | None = None) -> None:
        self.kwargs: dict = {}
        self.result = result or BatchResult(outcome="success")

    def __call__(self, **kwargs):
        self.kwargs = kwargs
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


class _FakeDeps:
    """Stands in for the composition root, so no real provider is constructed."""

    def __init__(self, **built_with) -> None:
        self.built_with = built_with
        self.skipped_sources = ["apify"]
        for stage in _STAGES:
            setattr(self, stage, object())


def _fake_build(**kwargs):
    return _FakeDeps(**kwargs)


@pytest.fixture
def invoke(tmp_path):
    """Drive run() with every real construction seam replaced."""

    def _invoke(argv, result=None, build=None):
        batch = _Recorder(result)
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
        )
        return code, batch, stdout.getvalue(), config_calls

    return _invoke


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
