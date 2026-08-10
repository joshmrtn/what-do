"""Writer for `run_history`, the only durable record of what a 2am run did.

The table has existed since the schema was written and nothing read or wrote
it. A row goes in when the batch starts and is updated when it ends, so a run
killed mid-flight still leaves evidence that it began — a row with a
`started_at` and no `completed_at` is a crash, which no end-of-run write could
ever record.
"""

from __future__ import annotations

import json
import sqlite3

from src.storage.db import connect
import uuid
from datetime import datetime
from pathlib import Path


def start_run(db_path: Path | str, started_at: datetime) -> str:
    """Record that a batch has begun, returning its run id.

    Args:
        db_path: Path to the SQLite database.
        started_at: When the run began.

    Returns:
        The new run's id, to be handed back to `finish_run`.
    """
    run_id = str(uuid.uuid4())
    conn = connect(db_path)
    try:
        conn.execute(
            "INSERT INTO run_history (id, started_at) VALUES (?, ?)",
            (run_id, started_at.isoformat()),
        )
        conn.commit()
    finally:
        conn.close()
    return run_id


def finish_run(
    db_path: Path | str,
    run_id: str,
    *,
    outcome: str,
    completed_at: datetime,
    stage_counts: dict[str, int] | None = None,
    errors: list[str] | None = None,
    skipped_sources: list[str] | None = None,
) -> None:
    """Complete a run's row with its outcome, counts, errors and skips.

    Skipped sources are stored apart from errors on purpose: a skip is a
    legitimate deployment state — no key for that source — and folding it in
    with failures would lose the distinction the credential policy rests on.

    An unknown `run_id` updates nothing rather than raising. The batch must
    never die trying to record that it died.

    Args:
        db_path: Path to the SQLite database.
        run_id: The id returned by `start_run`.
        outcome: One of `success`, `partial`, `failed`.
        completed_at: When the run ended. The duration is derived against the
            stored `started_at` rather than a passed-in value, so a resumed
            process still records the real elapsed time.
        stage_counts: Per-stage counts.
        errors: Stage failure messages.
        skipped_sources: Sources not built, normally for a missing credential.
    """
    conn = connect(db_path)
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
