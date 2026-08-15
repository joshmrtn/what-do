"""Deciding whether a refit is allowed to move the curve.

A **gate refuses** a bad move; a rate limit only slows one down. This is the
safety mechanism of the nightly refit, which is why it is deliberately dull:
fit on most of the corpus, score the incumbent and the candidate on rows
neither has seen, and keep the incumbent unless the candidate is better.

Smoothing is somebody else's job. This answers *whether*, not *how fast*.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from src.scoring.curve_fit import HUBER_DELTA, Observation, fit_curve

#: How the corpus is split. One fold is held out, the rest trains.
FOLDS = 5

#: Below this there is no honest way to hold rows back — the training side stops
#: being a corpus and the held-out side stops being evidence.
MINIMUM_ROWS_TO_GATE = 25


@dataclass(frozen=True)
class RefitDecision:
    """What the gate saw and what it concluded.

    Every field is here because `run_history` will want it: a stored score is
    only interpretable against the fit that produced it, and "the gate said no"
    is as much a part of that record as a move.
    """

    accepted: bool
    #: The parameters to adopt, or None when the incumbent stands.
    candidate: tuple[float, float] | None
    incumbent_score: float | None
    candidate_score: float | None
    train_rows: int
    holdout_rows: int
    reason: str | None = None


def assign_fold(event_id: str, folds: int = FOLDS) -> int:
    """Which fold a row belongs to, from its id alone.

    Hashed rather than drawn: the same corpus must split the same way on every
    run, or "any past score is recomputable from stored data" stops being true.
    It also means a row keeps its fold as the corpus grows, so adding tonight's
    events cannot reshuffle yesterday's split and silently change the verdict.

    `hashlib` rather than `hash()`, whose salt changes per interpreter.
    """
    digest = hashlib.sha256(event_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % folds


def _loss(rows: list[Observation], cap: float, saturation: float) -> float:
    """Mean Huber loss of a curve over rows — lower is better.

    The same loss the fit minimises, so the gate judges a candidate by the
    thing it was optimised for rather than by a second, differently-shaped
    opinion. Mean rather than sum, so train and held-out sizes are comparable.
    """
    chars = np.array([row.chars for row in rows], dtype=float)
    tags = np.array([row.tags for row in rows], dtype=float)
    residual = np.abs(tags - cap * (1.0 - np.exp(-chars / saturation)))
    return float(
        np.mean(
            np.where(
                residual <= HUBER_DELTA,
                0.5 * residual**2,
                HUBER_DELTA * (residual - 0.5 * HUBER_DELTA),
            )
        )
    )


def consider_refit(
    rows: list[Observation], *, incumbent: tuple[float, float]
) -> RefitDecision:
    """Whether tonight's corpus justifies moving off the incumbent curve.

    Fit on four fifths, score both curves on the fifth neither was fitted to,
    and adopt the candidate only if it does better there. The accepted
    parameters are then refitted over the **whole** corpus: the held-out rows
    are spent on the decision, not discarded from the answer.

    Args:
        rows: The regime's observations. Mixing regimes is the caller's error
            to avoid — this fits whatever it is given.
        incumbent: The `(cap, saturation)` currently in force.

    Returns:
        A decision carrying both scores and both row counts, whether or not it
        accepted. A refusal is as much a part of the record as a move.
    """
    if len(rows) < MINIMUM_ROWS_TO_GATE:
        return RefitDecision(
            accepted=False,
            candidate=None,
            incumbent_score=None,
            candidate_score=None,
            train_rows=len(rows),
            holdout_rows=0,
            reason=f"need at least {MINIMUM_ROWS_TO_GATE} rows to hold any out, got {len(rows)}",
        )

    holdout = [row for row in rows if assign_fold(row.event_id) == 0]
    train = [row for row in rows if assign_fold(row.event_id) != 0]

    if not holdout or len(train) < MINIMUM_ROWS_TO_GATE // 2:
        return RefitDecision(
            accepted=False,
            candidate=None,
            incumbent_score=None,
            candidate_score=None,
            train_rows=len(train),
            holdout_rows=len(holdout),
            reason="the split left too few rows on one side to judge",
        )

    trial = fit_curve(train)
    incumbent_score = _loss(holdout, *incumbent)
    candidate_score = _loss(holdout, *trial)

    if candidate_score >= incumbent_score:
        return RefitDecision(
            accepted=False,
            candidate=None,
            incumbent_score=incumbent_score,
            candidate_score=candidate_score,
            train_rows=len(train),
            holdout_rows=len(holdout),
            reason="the candidate did no better on rows it had not seen",
        )

    # Judged on four fifths, adopted from five. Returning the trial fit instead
    # would throw away a fifth of the evidence for no reason.
    return RefitDecision(
        accepted=True,
        candidate=fit_curve(rows),
        incumbent_score=incumbent_score,
        candidate_score=candidate_score,
        train_rows=len(train),
        holdout_rows=len(holdout),
    )
