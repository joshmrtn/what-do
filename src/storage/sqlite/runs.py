"""SQLite-backed `RunRepository`."""

from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from src.models.run import RunRecord
from src.storage.sqlite.connection import connect

_COLUMNS = (
    "id, started_at, completed_at, duration_ms, outcome, "
    "steps_completed, errors, skipped_sources, scoring_config, preference_revision_id"
)


def _to_record(row: tuple[Any, ...]) -> RunRecord:
    """Map one `run_history` row onto a RunRecord.

    Every nullable column is read defensively. A run that died before `finish`
    ran leaves the outcome, duration and JSON columns NULL — and a crashed run
    is precisely when this record is wanted, so the mapper cannot assume the
    row was ever completed.
    """
    return RunRecord(
        run_id=row[0],
        started_at=datetime.fromisoformat(row[1]),
        completed_at=datetime.fromisoformat(row[2]) if row[2] else None,
        duration_ms=row[3],
        outcome=row[4],
        stage_counts=json.loads(row[5]) if row[5] else {},
        errors=json.loads(row[6]) if row[6] else [],
        skipped_sources=json.loads(row[7]) if row[7] else [],
        scoring_config=row[8],
        preference_revision_id=row[9],
    )


class SqliteRunRepository:
    """Reads and writes `run_history` against a SQLite database."""

    def __init__(self, db_path: Path | str) -> None:
        self._db_path = db_path

    def start(
        self,
        started_at: datetime,
        scoring_config: str | None = None,
        dedup_config: str | None = None,
        preference_revision_id: str | None = None,
    ) -> str:
        """Record that a batch has begun, returning its run id."""
        run_id = str(uuid.uuid4())
        conn = connect(self._db_path)
        try:
            conn.execute(
                "INSERT INTO run_history "
                "(id, started_at, scoring_config, dedup_config, preference_revision_id) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    run_id,
                    started_at.isoformat(),
                    scoring_config,
                    dedup_config,
                    preference_revision_id,
                ),
            )
            conn.commit()
        finally:
            conn.close()
        return run_id

    def finish(
        self,
        run_id: str,
        *,
        outcome: str,
        completed_at: datetime,
        stage_counts: dict[str, int] | None = None,
        errors: list[str] | None = None,
        skipped_sources: list[str] | None = None,
    ) -> None:
        """Complete a run's row. An unknown id updates nothing rather than raising."""
        conn = connect(self._db_path)
        try:
            row = conn.execute(
                "SELECT started_at FROM run_history WHERE id = ?", (run_id,)
            ).fetchone()
            if row is None:
                return

            started_at = datetime.fromisoformat(row[0])
            duration_ms = int((completed_at - started_at).total_seconds() * 1000)

            conn.execute(
                "UPDATE run_history SET completed_at = ?, duration_ms = ?, "
                "steps_completed = ?, errors = ?, skipped_sources = ?, outcome = ? "
                "WHERE id = ?",
                (
                    completed_at.isoformat(),
                    duration_ms,
                    json.dumps(stage_counts or {}),
                    json.dumps(errors or []),
                    json.dumps(skipped_sources or []),
                    outcome,
                    run_id,
                ),
            )
            conn.commit()
        finally:
            conn.close()

    def open_run(self) -> RunRecord | None:
        """The most recent run that began and never finished, if any."""
        conn = connect(self._db_path)
        try:
            row = conn.execute(
                f"SELECT {_COLUMNS} FROM run_history "
                "WHERE completed_at IS NULL ORDER BY started_at DESC LIMIT 1"
            ).fetchone()
        finally:
            conn.close()
        return _to_record(row) if row is not None else None

    def latest(self) -> RunRecord | None:
        """The newest run row, finished or not.

        Deliberately not `open_run`'s counterpart. That one hunts for a crash
        and must ignore any successful run that came after it; this one answers
        *when did this system last do anything* — which is what `--status`
        reports when nothing is running.
        """
        conn = connect(self._db_path)
        try:
            row = conn.execute(
                f"SELECT {_COLUMNS} FROM run_history ORDER BY started_at DESC LIMIT 1"
            ).fetchone()
        finally:
            conn.close()
        return _to_record(row) if row is not None else None

    def get(self, run_id: str) -> RunRecord | None:
        """One run's record, or None if no such run exists."""
        conn = connect(self._db_path)
        try:
            row = conn.execute(
                f"SELECT {_COLUMNS} FROM run_history WHERE id = ?", (run_id,)
            ).fetchone()
        finally:
            conn.close()
        return _to_record(row) if row is not None else None
