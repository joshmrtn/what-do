"""Recommendation persistence and reload.

The batch's ordering is the product, so it is stored whole — every score
component, not just the total, and the rank itself. The CLI reads these rows
and renders them; it never re-derives an ordering of its own.

Storage splits what was one table into three, along the line between what was
*computed about an event* and *how it was ranked tonight*:

* `event_scores` — the semantic verdict, kept for every scored event
* `score_reasons` — each contribution, one row rather than a JSON blob
* `recommendations` — the ranking output alone

Nothing here derives a presentation label. The ordering is the product; how it
is displayed is the CLI's business alone.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

from src.models.event import Event
from src.models.recommendation import Recommendation
from src.scoring.similarity import Reason
from src.storage.db import connect
from src.storage.events import EVENT_COLUMNS, load_events, row_to_event

_SCORE_COLUMNS = "event_id, run_date, tag_score, summary_score, base_score, match"
_RANK_COLUMNS = "event_id, run_date, weather_adjustment, tag_confidence, final_score, rank"
_REASON_COLUMNS = (
    "event_id, run_date, position, factor, tag, matched_preference, "
    "similarity, contribution, direction"
)


def save_recommendations(
    recommendations: list[Recommendation], db_path: Path | str
) -> None:
    """Persist one run's scores, reasons and ranking, replacing its earlier attempt.

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

    run_dates = [(stamp,) for stamp in {r.run_date.isoformat() for r in recommendations}]

    conn = connect(db_path)
    try:
        # score_reasons and recommendations both cascade from event_scores, so
        # clearing the scores clears the run.
        conn.executemany("DELETE FROM event_scores WHERE run_date = ?", run_dates)

        conn.executemany(
            f"INSERT INTO event_scores ({_SCORE_COLUMNS}) VALUES (?, ?, ?, ?, ?, ?)",
            [
                (
                    r.event_id,
                    r.run_date.isoformat(),
                    None,
                    None,
                    r.base_score,
                    r.match,
                )
                for r in recommendations
            ],
        )
        conn.executemany(
            f"INSERT INTO recommendations ({_RANK_COLUMNS}) VALUES (?, ?, ?, ?, ?, ?)",
            [
                (
                    r.event_id,
                    r.run_date.isoformat(),
                    r.weather_adjustment,
                    r.tag_confidence,
                    r.final_score,
                    r.rank,
                )
                for r in recommendations
            ],
        )
        conn.executemany(
            f"INSERT INTO score_reasons ({_REASON_COLUMNS}) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    r.event_id,
                    r.run_date.isoformat(),
                    position,
                    reason.factor,
                    reason.tag,
                    reason.matched_preference,
                    reason.similarity,
                    reason.contribution,
                    reason.direction,
                )
                for r in recommendations
                for position, reason in enumerate(r.reasons)
            ],
        )
        conn.commit()
    finally:
        conn.close()


def _group_reasons(rows: list[tuple[Any, ...]]) -> dict[str, list[Reason]]:
    """Group reason rows by event, preserving position order."""
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


def load_recommendations(
    db_path: Path | str,
    run_date: date | None = None,
) -> list[Recommendation]:
    """Load persisted recommendations in rank order.

    Args:
        db_path: Path to the SQLite database.
        run_date: Restrict to one batch date. Defaults to every stored run —
            `load_ranked` is the one that narrows to the latest.

    Returns:
        Recommendations ordered by run date, then by the rank the batch assigned.
    """
    query = (
        "SELECT s.event_id, s.run_date, s.base_score, s.match, "
        "r.weather_adjustment, r.tag_confidence, r.final_score, r.rank "
        "FROM event_scores s JOIN recommendations r "
        "  ON r.event_id = s.event_id AND r.run_date = s.run_date "
    )
    stamp = run_date.isoformat() if run_date is not None else None

    conn = connect(db_path)
    try:
        if stamp is None:
            rows = conn.execute(query + "ORDER BY s.run_date, r.rank").fetchall()
            reason_rows = conn.execute(
                f"SELECT {_REASON_COLUMNS} FROM score_reasons ORDER BY event_id, position"
            ).fetchall()
        else:
            rows = conn.execute(
                query + "WHERE s.run_date = ? ORDER BY r.rank", (stamp,)
            ).fetchall()
            reason_rows = conn.execute(
                f"SELECT {_REASON_COLUMNS} FROM score_reasons WHERE run_date = ? "
                "ORDER BY event_id, position",
                (stamp,),
            ).fetchall()
    finally:
        conn.close()

    reasons = _group_reasons(reason_rows)
    return [_to_recommendation(row, reasons) for row in rows]


def _to_recommendation(
    row: tuple[Any, ...], reasons: dict[str, list[Reason]]
) -> Recommendation:
    """Rebuild a Recommendation from a joined score and ranking row."""
    event_id, run_date, base_score, match, weather, confidence, final_score, rank = row
    return Recommendation(
        recommendation_id=f"{run_date}:{event_id}",
        event_id=event_id,
        run_date=date.fromisoformat(run_date),
        base_score=base_score,
        weather_adjustment=weather,
        tag_confidence=confidence,
        final_score=final_score,
        match=match,
        rank=rank,
        reasons=reasons.get(event_id, []),
    )


def latest_run_date(db_path: Path | str) -> date | None:
    """The most recent batch date with stored recommendations, if any."""
    conn = connect(db_path)
    try:
        row = conn.execute("SELECT max(run_date) FROM recommendations").fetchone()
    finally:
        conn.close()

    return date.fromisoformat(row[0]) if row and row[0] else None


def load_ranked(
    db_path: Path | str,
    run_date: date | None = None,
) -> list[tuple[Recommendation, Event]]:
    """Load one run's recommendations alongside the events they rank.

    Composed from two reads rather than one join, so neither table has to know
    the other's row shape. A recommendation whose event has been purged is
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

    recommendations = load_recommendations(db_path, target)
    if not recommendations:
        return []

    events = {event.event_id: event for event in load_events(db_path)}
    return [(r, events[r.event_id]) for r in recommendations if r.event_id in events]


__all__ = [
    "EVENT_COLUMNS",
    "latest_run_date",
    "load_ranked",
    "load_recommendations",
    "row_to_event",
    "save_recommendations",
]
