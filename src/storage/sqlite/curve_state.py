"""The tag-confidence curve currently in force."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from src.storage.curve_state import CurveState
from src.storage.sqlite.connection import connect


class SqliteCurveStateRepository:
    """Reads and writes the single `curve_state` row."""

    def __init__(self, db_path: Path | str) -> None:
        self._db_path = db_path

    def load(self) -> CurveState | None:
        """The curve in force, or None when the config defaults still stand."""
        conn = connect(self._db_path)
        try:
            row = conn.execute(
                "SELECT cap, saturation, regime, updated_at, provenance "
                "FROM curve_state WHERE id = 1"
            ).fetchone()
        finally:
            conn.close()

        if row is None:
            return None
        return CurveState(
            cap=float(row[0]),
            saturation=float(row[1]),
            regime=row[2],
            updated_at=datetime.fromisoformat(row[3]),
            provenance=json.loads(row[4]) if row[4] else {},
        )

    def save(self, state: CurveState) -> None:
        """Replace the curve in force. One row, so this is an upsert on id 1."""
        conn = connect(self._db_path)
        try:
            conn.execute(
                "INSERT INTO curve_state (id, cap, saturation, regime, updated_at, provenance) "
                "VALUES (1, ?, ?, ?, ?, ?) "
                "ON CONFLICT(id) DO UPDATE SET cap = excluded.cap, "
                "saturation = excluded.saturation, regime = excluded.regime, "
                "updated_at = excluded.updated_at, provenance = excluded.provenance",
                (
                    state.cap,
                    state.saturation,
                    state.regime,
                    state.updated_at.isoformat(),
                    json.dumps(state.provenance, sort_keys=True, default=str),
                ),
            )
            conn.commit()
        finally:
            conn.close()
