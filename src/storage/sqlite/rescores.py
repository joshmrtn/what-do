"""SQLite-backed `RescoreRepository`."""

from __future__ import annotations

import sqlite3
import uuid
from datetime import date, datetime
from pathlib import Path
from typing import Any

from src.models.rescore import Rescore
from src.storage.sqlite.connection import connect

_COLUMNS = (
    "run_date, rescored_at, forecast_issued_at, preference_revision_id, events_rescored"
)


def _to_rescore(row: tuple[Any, ...]) -> Rescore:
    return Rescore(
        run_date=date.fromisoformat(row[0]),
        rescored_at=datetime.fromisoformat(row[1]),
        forecast_issued_at=datetime.fromisoformat(row[2]) if row[2] else None,
        preference_revision_id=row[3],
        events_rescored=row[4],
    )


class SqliteRescoreRepository:
    """Appends to `rescores` and reads a run's history back."""

    def __init__(self, db_path: Path | str) -> None:
        self._db_path = db_path

    def record(self, rescore: Rescore) -> None:
        """Append one rescore. Never an update — the previous rows stay."""
        conn = connect(self._db_path)
        try:
            conn.execute(
                f"INSERT INTO rescores (id, {_COLUMNS}) VALUES (?, ?, ?, ?, ?, ?)",
                (
                    str(uuid.uuid4()),
                    rescore.run_date.isoformat(),
                    rescore.rescored_at.isoformat(),
                    (
                        rescore.forecast_issued_at.isoformat()
                        if rescore.forecast_issued_at is not None
                        else None
                    ),
                    rescore.preference_revision_id,
                    rescore.events_rescored,
                ),
            )
            conn.commit()
        finally:
            conn.close()

    def latest_for(self, run_date: date) -> Rescore | None:
        """The most recent rescore of one run, or None if it has never been."""
        rows = self._select(run_date, limit=1)
        return _to_rescore(rows[0]) if rows else None

    def for_run(self, run_date: date) -> list[Rescore]:
        """Every rescore of one run, newest first."""
        return [_to_rescore(row) for row in self._select(run_date)]

    def _select(self, run_date: date, limit: int | None = None) -> list[Any]:
        conn = connect(self._db_path)
        try:
            return self._rows(conn, run_date, limit)
        finally:
            conn.close()

    @staticmethod
    def _rows(
        conn: sqlite3.Connection, run_date: date, limit: int | None
    ) -> list[Any]:
        sql = (
            f"SELECT {_COLUMNS} FROM rescores WHERE run_date = ? "
            "ORDER BY rescored_at DESC"
        )
        if limit is not None:
            sql += f" LIMIT {int(limit)}"
        return list(conn.execute(sql, (run_date.isoformat(),)).fetchall())
