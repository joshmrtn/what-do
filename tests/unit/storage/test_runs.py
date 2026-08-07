from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta, timezone

import pytest

from src.storage.db import init_db
from src.storage.runs import finish_run, start_run

STARTED = datetime(2026, 6, 15, 2, 0, 0, tzinfo=timezone.utc)


@pytest.fixture
def db(tmp_path):
    path = tmp_path / "batch.db"
    init_db(path)
    return path


def _row(db) -> dict:
    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute("SELECT * FROM run_history").fetchall()
    finally:
        conn.close()
    assert len(rows) == 1
    return dict(rows[0])


def test_a_started_run_is_recorded_before_it_finishes(db):
    """A run killed mid-flight leaves evidence it began; that is the whole point."""
    start_run(db, STARTED)

    row = _row(db)
    assert row["started_at"] == STARTED.isoformat()
    assert row["completed_at"] is None
    assert row["outcome"] is None


def test_each_run_gets_its_own_id(db):
    assert start_run(db, STARTED) != start_run(db, STARTED)


def test_finishing_records_the_outcome(db):
    run_id = start_run(db, STARTED)

    finish_run(db, run_id, outcome="partial", completed_at=STARTED + timedelta(minutes=3))

    assert _row(db)["outcome"] == "partial"


def test_finishing_derives_the_duration_from_the_stored_start(db):
    """Read back rather than passed in, so a resumed process still gets it right."""
    run_id = start_run(db, STARTED)

    finish_run(db, run_id, outcome="success", completed_at=STARTED + timedelta(seconds=90))

    assert _row(db)["duration_ms"] == 90_000


def test_stage_counts_are_stored(db):
    run_id = start_run(db, STARTED)

    finish_run(
        db,
        run_id,
        outcome="success",
        completed_at=STARTED,
        stage_counts={"ingested": 12, "ranked": 7},
    )

    assert json.loads(_row(db)["steps_completed"]) == {"ingested": 12, "ranked": 7}


def test_errors_are_stored(db):
    run_id = start_run(db, STARTED)

    finish_run(
        db, run_id, outcome="partial", completed_at=STARTED, errors=["enrichment failed: boom"]
    )

    assert json.loads(_row(db)["errors"]) == ["enrichment failed: boom"]


def test_skipped_sources_are_stored_apart_from_errors(db):
    """A skip is a deployment state, not a failure; conflating them loses that."""
    run_id = start_run(db, STARTED)

    finish_run(
        db, run_id, outcome="success", completed_at=STARTED, skipped_sources=["apify"]
    )

    row = _row(db)
    assert json.loads(row["skipped_sources"]) == ["apify"]
    assert json.loads(row["errors"]) == []


def test_finishing_an_unknown_run_is_not_an_error(db):
    """The batch must never die trying to record that it died."""
    finish_run(db, "no-such-run", outcome="failed", completed_at=STARTED)
