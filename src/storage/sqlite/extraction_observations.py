"""Append-only storage for extraction observations."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from src.storage.extraction_observations import ExtractionObservation
from src.storage.sqlite.connection import connect

_COLUMNS = (
    "event_id, observed_at, chars, tags, model, prompt_version, "
    "degradation, source, backfilled"
)


class SqliteExtractionObservationRepository:
    """Reads and appends `extraction_observations`."""

    def __init__(self, db_path: Path | str) -> None:
        self._db_path = db_path

    def append(self, observations: list[ExtractionObservation]) -> None:
        """Record extractions. Re-recording the same instant is a no-op.

        `INSERT OR IGNORE` on `(event_id, observed_at)`: a run that is retried,
        or an event saved twice within one run, must not double-count itself
        into the corpus.
        """
        if not observations:
            return
        conn = connect(self._db_path)
        try:
            conn.executemany(
                f"INSERT OR IGNORE INTO extraction_observations ({_COLUMNS}) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    (
                        o.event_id,
                        o.observed_at.isoformat(),
                        o.chars,
                        o.tags,
                        o.model,
                        o.prompt_version,
                        o.degradation,
                        o.source,
                        int(o.backfilled),
                    )
                    for o in observations
                ],
            )
            conn.commit()
        finally:
            conn.close()

    def load_all(self) -> list[ExtractionObservation]:
        """Every observation, oldest first — the chronology a detector needs."""
        conn = connect(self._db_path)
        try:
            rows = conn.execute(
                f"SELECT {_COLUMNS} FROM extraction_observations "
                "ORDER BY observed_at, event_id"
            ).fetchall()
        finally:
            conn.close()
        return [
            ExtractionObservation(
                event_id=r[0],
                observed_at=datetime.fromisoformat(r[1]),
                chars=int(r[2]),
                tags=int(r[3]),
                model=r[4],
                prompt_version=r[5],
                degradation=r[6],
                source=r[7],
                backfilled=bool(r[8]),
            )
            for r in rows
        ]
