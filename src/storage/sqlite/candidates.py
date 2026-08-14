"""SQLite-backed `CandidateRepository`."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from src.models.candidate_version import CandidateVersion
from src.models.event_candidate import EventCandidate
from src.storage.candidates import (
    CANDIDATE_COLUMNS,
    candidate_to_row,
    content_fingerprint,
    published_payload,
    row_to_candidate,
)
from src.storage.sqlite.connection import connect


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

    columns = CANDIDATE_COLUMNS.split(", ")
    placeholders = ", ".join("?" * len(columns))
    # An upsert rather than INSERT OR REPLACE, for one column's sake:
    # `discovered_at` is absent from the SET list, so a re-fetch cannot restamp
    # the first sighting. REPLACE is a delete and re-insert, which has no way to
    # keep anything from the row it replaces (#27).
    refreshed = ", ".join(
        f"{c} = excluded.{c}" for c in columns if c not in ("id", "discovered_at")
    )
    conn.executemany(
        f"INSERT INTO event_candidates ({CANDIDATE_COLUMNS}) "
        f"VALUES ({placeholders}) "
        f"ON CONFLICT(id) DO UPDATE SET {refreshed}",
        [candidate_to_row(c) for c in candidates],
    )
    # In the same statement batch as the row it describes, and after it, since
    # the version references it. Retaining what a source published is not a
    # separate call a caller could omit — that is what made overwriting the raw
    # layer possible in the first place.
    #
    # INSERT OR IGNORE against a (candidate_id, content_hash) key: an unchanged
    # republication collides with the version already stored and does nothing,
    # so the common case costs one no-op insert and the table grows only when
    # the text really moved.
    conn.executemany(
        "INSERT OR IGNORE INTO candidate_versions "
        "(candidate_id, content_hash, observed_at, payload) VALUES (?, ?, ?, ?)",
        [_version_row(c) for c in candidates],
    )


def _version_row(candidate: EventCandidate) -> tuple[str, str, str, str]:
    """A candidate's published content, keyed by its digest."""
    payload = published_payload(candidate)
    return (
        candidate.id,
        content_fingerprint(payload),
        (candidate.last_seen_at or candidate.discovered_at).isoformat(),
        json.dumps(payload, sort_keys=True, default=str),
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
        self, *, seen_since: datetime, starting_after: datetime
    ) -> list[EventCandidate]:
        """Candidates still in scope for a run, newest discovery last.

        Both bounds are canonicalised to UTC before the comparison. Stored
        timestamps are UTC and SQLite compares them as **text**, so a bound in
        any other zone compares a wall clock against an instant: measured on
        the live data, a local-form floor disagreed with the truth on 15
        candidates and a UTC-form floor on none.
        """
        conn = connect(self._db_path)
        try:
            rows = conn.execute(
                f"""SELECT {CANDIDATE_COLUMNS} FROM event_candidates
                    WHERE (start_time IS NOT NULL AND start_time >= ?)
                       OR (start_time IS NULL AND last_seen_at >= ?)
                    ORDER BY discovered_at, id""",
                (
                    starting_after.astimezone(timezone.utc).isoformat(),
                    seen_since.astimezone(timezone.utc).isoformat(),
                ),
            ).fetchall()
        finally:
            conn.close()

        return [row_to_candidate(row) for row in rows]

    def versions_for(self, candidate_id: str) -> list[CandidateVersion]:
        """Every distinct content this candidate has published, oldest first."""
        conn = connect(self._db_path)
        try:
            rows = conn.execute(
                "SELECT candidate_id, content_hash, observed_at, payload "
                "FROM candidate_versions WHERE candidate_id = ? "
                "ORDER BY observed_at, content_hash",
                (candidate_id,),
            ).fetchall()
        finally:
            conn.close()

        return [
            CandidateVersion(
                candidate_id=row[0],
                content_hash=row[1],
                observed_at=datetime.fromisoformat(row[2]),
                payload=json.loads(row[3]),
            )
            for row in rows
        ]
