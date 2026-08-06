"""Event candidate reload.

`IngestionService` persists candidates and returns only counts, so nothing could
read them back. The batch reads from here rather than passing objects through in
memory, which makes the ingest boundary crash-survivable: a re-run after a
failure picks up candidates already fetched without touching the network again.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

from src.models.event_candidate import EventCandidate

CANDIDATE_COLUMNS = (
    "id, source, source_type, url, image_url, raw_published_at, title, "
    "description, venue, location, start_time, end_time, discovered_at"
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
    conn = sqlite3.connect(db_path)
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
