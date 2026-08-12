"""Event candidate reload.

`IngestionService` persists candidates and returns only counts, so nothing could
read them back. The batch reads from here rather than passing objects through in
memory, which makes the ingest boundary crash-survivable: a re-run after a
failure picks up candidates already fetched without touching the network again.
"""

from __future__ import annotations

import json
import sqlite3

from src.storage.db import connect
from datetime import datetime
from pathlib import Path
from typing import Any

from src.models.event_candidate import EventCandidate
from src.models.tag import Tag

#: Shared by the reader and the writer below, so a new field cannot reach one
#: without the other. Issue #22: two hand-written column lists in two modules
#: drift silently, and the reader defaults the difference away.
CANDIDATE_COLUMNS = (
    "id, source, source_type, url, image_url, raw_published_at, title, "
    "description, venue, location, start_time, end_time, discovered_at, "
    "timing, summary, tags, metadata"
)


def row_to_candidate(row: tuple[Any, ...]) -> EventCandidate:
    """Rebuild an EventCandidate from a row of `CANDIDATE_COLUMNS`."""
    return EventCandidate(
        id=row[0],
        source=row[1],
        source_type=row[2],
        url=row[3],
        image_url=row[4],
        raw_published_at=_parse(row[5]),
        title=row[6],
        description=row[7],
        venue=row[8],
        location=row[9],
        start_time=_parse(row[10]),
        end_time=_parse(row[11]),
        discovered_at=datetime.fromisoformat(row[12]),
        timing=row[13],
        summary=row[14],
        tags=[Tag(text=t["tag"], weight=t["weight"]) for t in json.loads(row[15] or "[]")],
        metadata=json.loads(row[16] or "{}"),
    )


def candidate_to_row(candidate: EventCandidate) -> tuple[Any, ...]:
    """Flatten a candidate into a row of `CANDIDATE_COLUMNS`."""
    return (
        candidate.id,
        candidate.source,
        candidate.source_type,
        candidate.url,
        candidate.image_url,
        candidate.raw_published_at.isoformat() if candidate.raw_published_at else None,
        candidate.title,
        candidate.description,
        candidate.venue,
        candidate.location,
        candidate.start_time.isoformat() if candidate.start_time else None,
        candidate.end_time.isoformat() if candidate.end_time else None,
        candidate.discovered_at.isoformat(),
        candidate.timing,
        candidate.summary,
        json.dumps([{"tag": t.text, "weight": t.weight} for t in candidate.tags]),
        json.dumps(candidate.metadata),
    )


def save_candidates(
    candidates: list[EventCandidate], db_path: Path | str
) -> None:
    """Persist candidates, replacing any stored under the same id.

    Args:
        candidates: Candidates to store. Empty is a no-op.
        db_path: Path to the SQLite database.
    """
    if not candidates:
        return

    conn = connect(db_path)
    try:
        write_candidates(conn, candidates)
        conn.commit()
    finally:
        conn.close()


def write_candidates(
    conn: sqlite3.Connection, candidates: list[EventCandidate]
) -> None:
    """Write candidates on a caller's connection, without committing.

    Ingestion holds one connection across a whole fetch and commits once, so it
    needs the statement without the transaction around it.
    """
    placeholders = ", ".join("?" * len(CANDIDATE_COLUMNS.split(", ")))
    conn.executemany(
        f"INSERT OR REPLACE INTO event_candidates ({CANDIDATE_COLUMNS}) "
        f"VALUES ({placeholders})",
        [candidate_to_row(c) for c in candidates],
    )


def load_candidates(
    db_path: Path | str,
    discovered_since: datetime,
    starting_after: datetime,
) -> list[EventCandidate]:
    """Load candidates still in scope for a batch run.

    The window is a union, because either filter alone starves a source type.
    Social candidates carry no `start_time` at ingestion, so a forward-only
    filter drops all of them; a `discovered_at`-only filter eventually drops
    calendar events that are still upcoming.

    Args:
        db_path: Path to the SQLite database.
        discovered_since: Earliest `discovered_at` to accept, normally the
            lookback cutoff.
        starting_after: Earliest `start_time` to accept regardless of age,
            normally the run's now.

    Returns:
        Matching candidates, ordered by discovery then id. The order is fixed
        because dedup picks a merge base partly on the order it sees.
    """
    conn = connect(db_path)
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


def _parse(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value) if value else None
