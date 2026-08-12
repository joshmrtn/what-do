"""SQLite-backed `CandidateRepository`."""

from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path

from src.models.event_candidate import EventCandidate
from src.storage.candidates import (
    CANDIDATE_COLUMNS,
    candidate_to_row,
    row_to_candidate,
)
from src.storage.db import connect


def write_candidates(
    conn: sqlite3.Connection, candidates: list[EventCandidate]
) -> None:
    """Write candidates on a caller's connection, without committing.

    Ingestion holds one connection across a whole fetch and commits once,
    because the inserts hold a RESERVED lock that discovery — which opens its
    own connection — would otherwise wait on until SQLite's timeout. So the
    statement is available apart from the transaction around it.
    """
    if not candidates:
        return

    placeholders = ", ".join("?" * len(CANDIDATE_COLUMNS.split(", ")))
    conn.executemany(
        f"INSERT OR REPLACE INTO event_candidates ({CANDIDATE_COLUMNS}) "
        f"VALUES ({placeholders})",
        [candidate_to_row(c) for c in candidates],
    )


class SqliteCandidateRepository:
    """Reads and writes `event_candidates`."""

    def __init__(self, db_path: Path | str) -> None:
        self._db_path = db_path

    def save(self, candidates: list[EventCandidate]) -> None:
        """Insert candidates, replacing any stored under the same id."""
        if not candidates:
            return

        conn = connect(self._db_path)
        try:
            write_candidates(conn, candidates)
            conn.commit()
        finally:
            conn.close()

    def for_window(
        self, *, discovered_since: datetime, starting_after: datetime
    ) -> list[EventCandidate]:
        """Candidates still in scope for a run, newest discovery last."""
        conn = connect(self._db_path)
        try:
            rows = conn.execute(
                f"""SELECT {CANDIDATE_COLUMNS} FROM event_candidates
                    WHERE discovered_at >= ?
                       OR (start_time IS NOT NULL AND start_time >= ?)
                    ORDER BY discovered_at, id""",
                (discovered_since.isoformat(), starting_after.isoformat()),
            ).fetchall()
        finally:
            conn.close()

        return [row_to_candidate(row) for row in rows]
