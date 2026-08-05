"""Recommendation persistence and reload.

The batch's ordering is the product, so it is stored whole — every score
component, not just the total, and the rank itself. The CLI reads these rows
and renders them; it never re-derives an ordering of its own.
"""

from __future__ import annotations

import sqlite3
from datetime import date
from pathlib import Path
from typing import Any

from src.models.event import Event
from src.models.recommendation import Recommendation, reasons_from_json, reasons_to_json
from src.storage.events import EVENT_COLUMNS, row_to_event

_COLUMNS = (
    "id, event_id, run_date, base_score, weather_adjustment, tag_confidence, "
    "final_score, tier, match, rank, reasons"
)


def _qualify(columns: str, alias: str) -> str:
    """Prefix a column list with a table alias, for use in a join."""
    return ", ".join(f"{alias}.{name}" for name in columns.split(", "))


def recommendation_to_row(recommendation: Recommendation) -> tuple[Any, ...]:
    """Flatten a Recommendation into a row for the recommendations table."""
    return (
        recommendation.recommendation_id,
        recommendation.event_id,
        recommendation.run_date.isoformat(),
        recommendation.base_score,
        recommendation.weather_adjustment,
        recommendation.tag_confidence,
        recommendation.final_score,
        recommendation.tier,
        recommendation.match,
        recommendation.rank,
        reasons_to_json(recommendation.reasons),
    )


def row_to_recommendation(row: tuple[Any, ...]) -> Recommendation:
    """Rebuild a Recommendation from a row selected with _COLUMNS."""
    return Recommendation(
        recommendation_id=row[0],
        event_id=row[1],
        run_date=date.fromisoformat(row[2]),
        base_score=row[3],
        weather_adjustment=row[4],
        tag_confidence=row[5],
        final_score=row[6],
        tier=row[7],
        match=row[8],
        rank=row[9],
        reasons=reasons_from_json(row[10]),
    )


def save_recommendations(
    recommendations: list[Recommendation], db_path: Path | str
) -> None:
    """Persist one run's recommendations, replacing any previous rows for its dates.

    A re-run of the same date supersedes its earlier attempt rather than
    accumulating a second copy — otherwise a batch retried after a partial
    failure would leave the CLI reading two conflicting orderings. Replacement
    is scoped to the run dates being written, so previous nights are untouched.

    Args:
        recommendations: The run's ranked output. Empty is a no-op, not an
            instruction to clear the table.
        db_path: Path to the SQLite database.
    """
    if not recommendations:
        return

    placeholders = ", ".join("?" * len(_COLUMNS.split(", ")))
    run_dates = {r.run_date.isoformat() for r in recommendations}

    conn = sqlite3.connect(db_path)
    try:
        conn.executemany(
            "DELETE FROM recommendations WHERE run_date = ?",
            [(run_date,) for run_date in run_dates],
        )
        conn.executemany(
            f"INSERT INTO recommendations ({_COLUMNS}) VALUES ({placeholders})",
            [recommendation_to_row(r) for r in recommendations],
        )
        conn.commit()
    finally:
        conn.close()


def load_recommendations(
    db_path: Path | str, run_date: date | None = None
) -> list[Recommendation]:
    """Load persisted recommendations in rank order.

    Args:
        db_path: Path to the SQLite database.
        run_date: Restrict to one batch date. Defaults to every stored run.

    Returns:
        Recommendations ordered by run date, then by the rank the batch assigned.
    """
    query = f"SELECT {_COLUMNS} FROM recommendations"
    params: tuple[Any, ...] = ()
    if run_date is not None:
        query += " WHERE run_date = ?"
        params = (run_date.isoformat(),)
    query += " ORDER BY run_date, rank"

    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(query, params).fetchall()
    finally:
        conn.close()

    return [row_to_recommendation(row) for row in rows]


def latest_run_date(db_path: Path | str) -> date | None:
    """Return the most recent run date held in the recommendations table.

    Previous runs are kept deliberately, so a reader that wants "the current
    ordering" has to ask which run that is rather than reading the whole table.

    Returns:
        The latest run date, or None if no batch has ranked anything yet.
    """
    conn = sqlite3.connect(db_path)
    try:
        row = conn.execute("SELECT MAX(run_date) FROM recommendations").fetchone()
    finally:
        conn.close()

    return date.fromisoformat(row[0]) if row and row[0] else None


def load_ranked(
    db_path: Path | str, run_date: date | None = None
) -> list[tuple[Recommendation, Event]]:
    """Load one run's recommendations alongside the events they rank.

    The join is inner, so a recommendation whose event has been purged is
    skipped rather than surfacing as a half-empty row — one missing event must
    not take down the whole view.

    Args:
        db_path: Path to the SQLite database.
        run_date: Which batch to read. Defaults to the latest one.

    Returns:
        (recommendation, event) pairs in the rank order the batch assigned.
        Empty if nothing has been ranked yet.
    """
    target = run_date if run_date is not None else latest_run_date(db_path)
    if target is None:
        return []

    split = len(_COLUMNS.split(", "))
    query = (
        f"SELECT {_qualify(_COLUMNS, 'r')}, {_qualify(EVENT_COLUMNS, 'e')} "
        "FROM recommendations r JOIN events e ON e.id = r.event_id "
        "WHERE r.run_date = ? ORDER BY r.rank"
    )

    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(query, (target.isoformat(),)).fetchall()
    finally:
        conn.close()

    return [(row_to_recommendation(row[:split]), row_to_event(row[split:])) for row in rows]
