"""Running the nightly refit and recording what it concluded.

The composition step: everything below it is pure. This reads events, builds
observations, plans the move, and writes the result for the *next* run to score
with — never this one, so a night is not scored with constants that moved under
it.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from src.models.event import Event
from src.scoring.change_detection import detect_change
from src.scoring.curve_fit import Observation
from src.scoring.refit_policy import RefitPolicy, plan_refit
from src.scoring.source_terms import source_multipliers
from src.storage.curve_state import CurveState


def observations(events: list[Event]) -> list[Observation]:
    """The rows a refit may learn from, in creation order.

    Only extractions with provenance and a stored input. Rows without
    `extraction_model` predate provenance and cannot be told apart from a
    different prompt's output; rows without `extraction_input_chars` predate the
    observation being kept and their length would be a reconstruction.

    Keyed on `source` rather than `source_type`: the category holds feeds with
    genuinely different characteristics, so a term keyed on it averages
    populations that describe neither (#34).

    Chronological, because the change detector reads them as a series.
    """
    usable = [
        event
        for event in events
        if event.extraction_model is not None
        and event.extraction_input_chars
        and event.superseded_by is None
    ]
    usable.sort(key=lambda event: event.created_at)
    return [
        Observation(
            event_id=event.event_id,
            chars=event.extraction_input_chars or 0,
            tags=float(len(event.tags)),
            source_type=event.source or event.source_type,
            regime=f"{event.extraction_model}/{event.extraction_prompt_version}",
        )
        for event in usable
    ]


def run_refit(
    events: list[Event],
    *,
    incumbent: tuple[float, float],
    now: datetime,
    policy: RefitPolicy | None = None,
) -> CurveState | None:
    """Tonight's refit, or None when there is nothing to record.

    Returns a state even when the gate refuses, because a refusal carrying its
    scores is part of the record — the only case that records nothing is having
    no usable rows at all.
    """
    rows = observations(events)
    if not rows:
        return None

    outcome = plan_refit(rows, incumbent=incumbent, policy=policy)
    in_regime = [row for row in rows if row.regime == outcome.regime]

    provenance: dict[str, Any] = {
        "regime": outcome.regime,
        "rows": len(in_regime),
        "accepted": outcome.decision.accepted,
        "reason": outcome.decision.reason,
        "train_rows": outcome.decision.train_rows,
        "holdout_rows": outcome.decision.holdout_rows,
        "incumbent_score": outcome.decision.incumbent_score,
        "candidate_score": outcome.decision.candidate_score,
        "fitted": outcome.decision.candidate,
        "incumbent": list(incumbent),
        "applied": list(outcome.parameters),
    }
    if outcome.decision.accepted:
        # Only meaningful against a curve worth having: multipliers and change
        # points computed off an unarmed regime's incumbent would describe a
        # curve nothing was fitted to.
        provenance["source_multipliers"] = source_multipliers(
            in_regime, *outcome.parameters
        )
        provenance["change_points"] = detect_change(in_regime, *outcome.parameters)

    return CurveState(
        cap=outcome.parameters[0],
        saturation=outcome.parameters[1],
        regime=outcome.regime,
        updated_at=now,
        provenance=provenance,
    )
