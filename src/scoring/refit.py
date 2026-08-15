"""Running the nightly refit and recording what it concluded.

The composition step: everything below it is pure. This reads events, builds
observations, plans the move, and writes the result for the *next* run to score
with — never this one, so a night is not scored with constants that moved under
it.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from src.scoring.change_detection import detect_change
from src.scoring.curve_fit import Observation
from src.scoring.refit_policy import RefitPolicy, plan_refit
from src.storage.extraction_observations import ExtractionObservation
from src.scoring.source_terms import source_multipliers
from src.storage.curve_state import CurveState


def observations(recorded: list[ExtractionObservation]) -> list[Observation]:
    """The rows a refit may learn from, in the order they were observed.

    Read from the extraction log rather than from `events`, and that is the
    whole point: `events` keeps only the latest extraction, so a corpus read
    there is current state, and its only usable timestamp is `created_at` —
    when the *event* was created. Every row in the largest feed was
    re-extracted days after creation, so a series ordered that way is not a
    chronology of extractions at all, and the change detector was reading one.

    Only observations carrying provenance: without `model` a row cannot be told
    apart from a different prompt's output.

    Keyed on the feed rather than the category, which averages populations that
    describe neither (#34).
    """
    usable = [row for row in recorded if row.model is not None and row.chars > 0]
    usable.sort(key=lambda row: (row.observed_at, row.event_id))
    return [
        Observation(
            event_id=row.event_id,
            chars=row.chars,
            tags=float(row.tags),
            source_type=row.source or "unknown",
            regime=f"{row.model}/{row.prompt_version}",
        )
        for row in usable
    ]


def run_refit(
    recorded: list[ExtractionObservation],
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
    rows = observations(recorded)
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
