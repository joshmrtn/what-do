"""Reader for the discovered-entity table.

Venue discovery promotes handles to `active` in `candidate_entities`, and
`IngestionService` syncs `seeds.yaml` into the same table at depth 0. Nothing
read the active set back, so discovery could never feed the next night's
ingestion — the social adapters take their handles at construction, and the
composition root is what needs them.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from src.storage.db import connect, has_schema


def load_active_handles(db_path: Path | str) -> list[str]:
    """Return every handle currently active for ingestion, alphabetically.

    Args:
        db_path: Path to the SQLite database.

    Returns:
        Active handles, or an empty list if the database has no schema yet —
        a first run, before any batch has initialised it.
    """
    if not has_schema(db_path):
        return []

    conn = connect(db_path)
    try:
        rows = conn.execute(
            "SELECT handle FROM candidate_entities WHERE state = 'active' "
            "ORDER BY handle"
        ).fetchall()
    finally:
        conn.close()

    return [row[0] for row in rows]
