"""SQLite-backed `ScoreRepository`."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

from src.models.event_score import EventScore
from src.scoring.similarity import Reason
from src.storage.db import connect

_SCORE_COLUMNS = (
    "event_id, run_date, tag_score, summary_score, base_score, tag_confidence, match"
)
_REASON_COLUMNS = (
    "event_id, run_date, position, factor, tag, matched_preference, "
    "similarity, contribution, direction"
)


def _group_reasons(rows: list[tuple[Any, ...]]) -> dict[str, list[Reason]]:
    """Group reason rows by event, preserving the stored position order."""
    grouped: dict[str, list[Reason]] = {}
    for event_id, _, _, factor, tag, matched, similarity, contribution, direction in rows:
        grouped.setdefault(event_id, []).append(
            Reason(
                factor=factor,
                matched_preference=matched,
                similarity=similarity,
                contribution=contribution,
                direction=direction,
                tag=tag,
            )
        )
    return grouped


class SqliteScoreRepository:
    """Reads and writes `event_scores` and `score_reasons`."""

    def __init__(self, db_path: Path | str) -> None:
        self._db_path = db_path

    def save(self, scores: list[EventScore]) -> None:
        """Insert scores for one or more run dates, replacing those runs."""
        if not scores:
            return

        run_dates = [(stamp,) for stamp in {s.run_date.isoformat() for s in scores}]

        conn = connect(self._db_path)
        try:
            # score_reasons and rankings both cascade from event_scores, so
            # clearing the scores clears the run.
            conn.executemany("DELETE FROM event_scores WHERE run_date = ?", run_dates)
            conn.executemany(
                f"INSERT INTO event_scores ({_SCORE_COLUMNS}) VALUES (?, ?, ?, ?, ?, ?, ?)",
                [
                    (
                        s.event_id,
                        s.run_date.isoformat(),
                        s.tag_score,
                        s.summary_score,
                        s.base_score,
                        s.tag_confidence,
                        s.match,
                    )
                    for s in scores
                ],
            )
            conn.executemany(
                f"INSERT INTO score_reasons ({_REASON_COLUMNS}) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    (
                        s.event_id,
                        s.run_date.isoformat(),
                        position,
                        reason.factor,
                        reason.tag,
                        reason.matched_preference,
                        reason.similarity,
                        reason.contribution,
                        reason.direction,
                    )
                    for s in scores
                    for position, reason in enumerate(s.reasons)
                ],
            )
            conn.commit()
        finally:
            conn.close()

    def for_run(self, run_date: date) -> list[EventScore]:
        """Every score stored for one batch date, with its reasons reattached."""
        stamp = run_date.isoformat()
        conn = connect(self._db_path)
        try:
            rows = conn.execute(
                f"SELECT {_SCORE_COLUMNS} FROM event_scores WHERE run_date = ?", (stamp,)
            ).fetchall()
            reason_rows = conn.execute(
                f"SELECT {_REASON_COLUMNS} FROM score_reasons WHERE run_date = ? "
                "ORDER BY event_id, position",
                (stamp,),
            ).fetchall()
        finally:
            conn.close()

        reasons = _group_reasons(reason_rows)
        return [
            EventScore(
                event_id=row[0],
                run_date=date.fromisoformat(row[1]),
                tag_score=row[2],
                summary_score=row[3],
                base_score=row[4],
                tag_confidence=row[5],
                match=row[6],
                reasons=reasons.get(row[0], []),
            )
            for row in rows
        ]
