"""What the refit may do once the gate has spoken.

Four separate jobs, kept separate on purpose: **arming** decides when to start,
**regimes** decide what may be fitted together, **bounds** refuse the
impossible, and **EWMA** decides how fast. Whether to move at all is the gate's
question and is answered in `refit_gate`.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from src.scoring.curve_fit import Observation
from src.scoring.refit_gate import RefitDecision, consider_refit

#: Rows a regime needs before it is fitted at all.
#:
#: Measured 2026-08-14: two independent halves of a 273-row corpus fitted to
#: cap 3.90/3.65 and saturation 135/115 against a full-corpus 3.80/127, so ~135
#: rows already replicates. 200 is that with margin. The original reasoning —
#: "N≈1000, weeks of data" — was conservative by about four times.
ARMING_ROWS = 200

#: A sanity rail, not a rate limit: the values beyond which a parameter stops
#: meaning anything. Deliberately wide, so a genuinely changed model or a new
#: source can move the fit a long way without being refused.
CAP_BOUNDS = (1.0, 12.0)
SATURATION_BOUNDS = (20.0, 2000.0)

#: How much of an accepted move lands per night. 0.15 is a ~13-night half-life,
#: so a full swing arrives over roughly a fortnight.
DEFAULT_ALPHA = 0.15


@dataclass(frozen=True)
class RefitPolicy:
    """The pace and the rails. Separated so a caller can reason about one."""

    arming_rows: int = ARMING_ROWS
    alpha: float = DEFAULT_ALPHA
    cap_bounds: tuple[float, float] = CAP_BOUNDS
    saturation_bounds: tuple[float, float] = SATURATION_BOUNDS


@dataclass(frozen=True)
class RefitOutcome:
    """What tonight's refit concluded, and what the next run should use."""

    #: The parameters to score with — the incumbent unless a move was accepted
    #: *and* survived the bounds, in which case a smoothed step toward it.
    parameters: tuple[float, float]
    decision: RefitDecision
    regime: str | None


def within_bounds(
    cap: float,
    saturation: float,
    policy: RefitPolicy | None = None,
) -> bool:
    """Whether a fitted pair is worth taking seriously at all."""
    policy = policy or RefitPolicy()
    return (
        policy.cap_bounds[0] <= cap <= policy.cap_bounds[1]
        and policy.saturation_bounds[0] <= saturation <= policy.saturation_bounds[1]
    )


def smooth(old: float, new: float, alpha: float = DEFAULT_ALPHA) -> float:
    """One night's step from `old` toward `new`.

    Exponential rather than a percentage clamp: no cliff, and it is the standard
    answer. The clamp it replaces was the discrete form of this, and given the
    gate its only remaining job was bounding a run of individually-plausible
    moves — which the domain bounds do better.
    """
    return (1.0 - alpha) * old + alpha * new


def plan_refit(
    rows: list[Observation],
    *,
    incumbent: tuple[float, float],
    regime: str | None = None,
    policy: RefitPolicy | None = None,
) -> RefitOutcome:
    """Decide what the next run should score with.

    Args:
        rows: Every observation available, across all regimes.
        incumbent: The `(cap, saturation)` currently in force.
        regime: Which regime to fit. Defaults to the one with the most rows,
            which is the live one on any ordinary night.
        policy: Pace and rails.

    Returns:
        The parameters to use and the reasoning behind them — including when
        nothing moved, since a refusal is as much a part of the record as a
        move.
    """
    policy = policy or RefitPolicy()

    if not rows:
        return RefitOutcome(incumbent, _held("no rows", 0), None)

    if regime is None:
        regime = Counter(row.regime for row in rows).most_common(1)[0][0]
    in_regime = [row for row in rows if row.regime == regime]

    # Arming is checked before the gate, not inside it: a regime with too few
    # rows has nothing to judge, and a fresh deployment sits here for weeks
    # without that being an error.
    if len(in_regime) < policy.arming_rows:
        return RefitOutcome(
            incumbent,
            _held(
                f"regime {regime!r} has {len(in_regime)} rows, "
                f"below the {policy.arming_rows} needed to arm",
                len(in_regime),
            ),
            regime,
        )

    decision = consider_refit(in_regime, incumbent=incumbent)
    if not decision.accepted or decision.candidate is None:
        return RefitOutcome(incumbent, decision, regime)

    if not within_bounds(*decision.candidate, policy):
        # Refused outright rather than smoothed toward: a value outside the
        # domain is not a move to make slowly, it is one not to make.
        return RefitOutcome(
            incumbent,
            RefitDecision(
                accepted=False,
                candidate=None,
                incumbent_score=decision.incumbent_score,
                candidate_score=decision.candidate_score,
                train_rows=decision.train_rows,
                holdout_rows=decision.holdout_rows,
                reason=f"fit {decision.candidate} is outside the sane domain",
            ),
            regime,
        )

    return RefitOutcome(
        (
            smooth(incumbent[0], decision.candidate[0], policy.alpha),
            smooth(incumbent[1], decision.candidate[1], policy.alpha),
        ),
        decision,
        regime,
    )


def _held(reason: str, rows: int) -> RefitDecision:
    """A decision that nothing moved, carrying why."""
    return RefitDecision(
        accepted=False,
        candidate=None,
        incumbent_score=None,
        candidate_score=None,
        train_rows=rows,
        holdout_rows=0,
        reason=reason,
    )
