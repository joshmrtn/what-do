"""Recommendation data model — one scored, ranked event for a single batch run."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date

from src.scoring.similarity import Reason


@dataclass(frozen=True)
class Recommendation:
    """One scored, ranked event for a single batch run.

    Frozen because it records what a run decided. `rank` is stored rather than
    recomputed at read time, so a reader cannot accidentally reorder the batch's
    decision, and a later "why did this move?" is answerable from history.

    Args:
        recommendation_id: Deterministic id derived from run_date and event_id.
        event_id: The event this ranks.
        run_date: The batch date that produced it.
        base_score: Semantic score from the similarity engine, before anything else.
        weather_adjustment: Signed comfort adjustment; 0.0 unless the event is outdoors.
        tag_confidence: How much of the expected tag count the extraction produced.
        final_score: What the ordering is actually built on.
        match: "yes", "maybe", or "no" — the advisory classification.
        rank: 1-based position within this run.
        reasons: Every contribution that produced final_score.
    """

    recommendation_id: str
    event_id: str
    run_date: date
    base_score: float
    weather_adjustment: float
    tag_confidence: float
    final_score: float
    match: str
    rank: int
    reasons: list[Reason] = field(default_factory=list)


def make_recommendation_id(run_date: date, event_id: str) -> str:
    """Build the id for one event's recommendation on one run date.

    Deterministic rather than random: two runs of the same batch over the same
    events must be identical, and a uuid4 would make that untestable and would
    silently accumulate duplicate rows for the same decision.
    """
    return f"{run_date.isoformat()}:{event_id}"


def reasons_to_json(reasons: list[Reason]) -> str:
    """Serialise reasons for the recommendations.reasons column."""
    return json.dumps(
        [
            {
                "factor": r.factor,
                "matched_preference": r.matched_preference,
                "similarity": r.similarity,
                "contribution": r.contribution,
                "direction": r.direction,
                "tag": r.tag,
            }
            for r in reasons
        ]
    )


def reasons_from_json(raw: str) -> list[Reason]:
    """Deserialise the recommendations.reasons column."""
    return [
        Reason(
            factor=entry["factor"],
            matched_preference=entry["matched_preference"],
            similarity=entry["similarity"],
            contribution=entry["contribution"],
            direction=entry["direction"],
            tag=entry.get("tag"),
        )
        for entry in json.loads(raw)
    ]
