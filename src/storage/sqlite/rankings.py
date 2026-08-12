"""SQLite-backed `RankingRepository`."""

from __future__ import annotations

from datetime import date
from pathlib import Path

from src.models.ranking import Ranking
from src.storage.sqlite.connection import connect

_COLUMNS = "event_id, run_date, weather_adjustment, final_score, rank"


class SqliteRankingRepository:
    """Reads and writes `rankings`."""

    def __init__(self, db_path: Path | str) -> None:
        self._db_path = db_path

    def save(self, rankings: list[Ranking]) -> None:
        """Insert placements for one or more run dates, replacing those runs."""
        if not rankings:
            return

        run_dates = [(stamp,) for stamp in {r.run_date.isoformat() for r in rankings}]

        conn = connect(self._db_path)
        try:
            conn.executemany("DELETE FROM rankings WHERE run_date = ?", run_dates)
            conn.executemany(
                f"INSERT INTO rankings ({_COLUMNS}) VALUES (?, ?, ?, ?, ?)",
                [
                    (
                        r.event_id,
                        r.run_date.isoformat(),
                        r.weather_adjustment,
                        r.final_score,
                        r.rank,
                    )
                    for r in rankings
                ],
            )
            conn.commit()
        finally:
            conn.close()

    def for_run(self, run_date: date) -> list[Ranking]:
        """One batch date's placements, in the rank the batch assigned."""
        conn = connect(self._db_path)
        try:
            rows = conn.execute(
                f"SELECT {_COLUMNS} FROM rankings WHERE run_date = ? ORDER BY rank",
                (run_date.isoformat(),),
            ).fetchall()
        finally:
            conn.close()

        return [
            Ranking(
                event_id=row[0],
                run_date=date.fromisoformat(row[1]),
                weather_adjustment=row[2],
                final_score=row[3],
                rank=row[4],
            )
            for row in rows
        ]

    def latest_run_date(self) -> date | None:
        """The most recent batch date that produced a ranking, if any."""
        conn = connect(self._db_path)
        try:
            row = conn.execute("SELECT max(run_date) FROM rankings").fetchone()
        finally:
            conn.close()

        return date.fromisoformat(row[0]) if row and row[0] else None
