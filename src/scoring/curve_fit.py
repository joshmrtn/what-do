"""Fitting the tag-confidence curve to observed extractions.

Pure: observations in, two constants out. No I/O, no clock, no config. Arming,
the held-out gate, smoothing and per-source terms are separate jobs and live
elsewhere — conflating them is how a refit becomes either inert or wild.

The shipped constants were fitted over 24 events driven through Gemini. Measured
against 273 real rows on 2026-08-14 they score R² 0.465, where a refit scores
0.675: the local model produces fewer tags than that fit expects, so the curve
is demonstrably wrong for the population it is applied to.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

#: Below this a fit is not a fit. Arming is a separate decision with a much
#: higher bar; this only stops three points looking like a curve.
MINIMUM_ROWS = 10

#: Huber threshold, in tags. Residuals beyond it are penalised linearly rather
#: than quadratically, so one strange listing cannot dominate. A 5,000-character
#: event earning a single tag is a real possibility and has enormous leverage on
#: `cap` under plain least squares.
HUBER_DELTA = 1.5

#: Search bounds for the fit itself. Wider than the domain bounds a caller
#: should accept — this is where a value stops being computable, not where it
#: stops being sensible.
_CAP_RANGE = (0.5, 15.0)
_SATURATION_RANGE = (10.0, 3000.0)


@dataclass(frozen=True)
class Observation:
    """One extraction, as the fit sees it.

    `chars` is the length of the text the model was actually asked, stored at
    extraction time rather than reconstructed — a rebuilt input is wrong the
    moment the builder or the event's fields move, and both do.
    """

    #: The event this came from. Carried so held-out folds can be assigned by
    #: hashing it — the same corpus must split the same way on every run.
    event_id: str
    chars: int
    tags: float
    #: The **feed**, `event_candidates.source` — not `source_type`, which is the
    #: category above it and holds two feeds for `northshorenightout` alone,
    #: with 290 vs 83 mean chars (#34). Named for what it holds: while this was
    #: called `source_type` and populated with the feed, `source_type` meant
    #: "feed" here and "category" everywhere else, and the id-churn measurement
    #: reached for the wrong one.
    source: str
    #: Which extraction regime produced this row — model and prompt version.
    #: Rows from two regimes are never fitted together: a prompt or model change
    #: is a step, not drift, and a curve fitted across one describes a
    #: population that never existed.
    regime: str = "unknown"


def expected_tags(chars: float, cap: float, saturation: float) -> float:
    """How many tags an input of this length typically earns.

    Saturating exponential: rises from zero and approaches `cap`. Monotonic in
    `chars` by construction, which is the one property any model whose only
    input is a length must have.

    **This measures conformity to what the extractor usually does, not whether
    a tag is correct.** An event the model reliably mis-tags looks perfectly
    confident. Only a labelled sample can speak to correctness.
    """
    return float(cap * (1.0 - math.exp(-chars / saturation)))


def fit_curve(rows: list[Observation]) -> tuple[float, float]:
    """The `(cap, saturation)` best describing these observations.

    Deterministic and order-independent: the same corpus must fit identically on
    two runs, or "any past score is recomputable from stored data" stops being
    true.

    Args:
        rows: Observations to fit. Every one needs a positive `chars`.

    Returns:
        `(cap, saturation)`, rounded so the result is stable across platforms
        and readable in a log line.

    Raises:
        ValueError: If there are too few rows, or any row has no length. A fit
            over three points is not a fit and must not look like one.
    """
    if len(rows) < MINIMUM_ROWS:
        raise ValueError(f"need at least {MINIMUM_ROWS} rows to fit, got {len(rows)}")
    if any(row.chars <= 0 for row in rows):
        raise ValueError("every observation needs a positive chars")

    chars = np.array([row.chars for row in rows], dtype=float)
    tags = np.array([row.tags for row in rows], dtype=float)

    def loss(cap: float, saturation: float) -> float:
        residual = np.abs(tags - cap * (1.0 - np.exp(-chars / saturation)))
        # Huber: quadratic near zero, linear beyond, so leverage is bounded.
        return float(
            np.sum(
                np.where(
                    residual <= HUBER_DELTA,
                    0.5 * residual**2,
                    HUBER_DELTA * (residual - 0.5 * HUBER_DELTA),
                )
            )
        )

    # Coarse sweep then successive refinement. The surface is smooth and
    # two-dimensional, so this is both cheaper and more predictable than a
    # gradient method with a starting point to argue about — and it has no
    # random component, which the determinism requirement rules out anyway.
    best = (float("inf"), 0.0, 0.0)
    for cap in np.linspace(*_CAP_RANGE, 60):
        for saturation in np.geomspace(*_SATURATION_RANGE, 60):
            value = loss(cap, saturation)
            if value < best[0]:
                best = (value, float(cap), float(saturation))

    _, cap, saturation = best
    for span in (0.5, 0.1, 0.02):
        cap_lo, cap_hi = max(_CAP_RANGE[0], cap - span * 4), min(_CAP_RANGE[1], cap + span * 4)
        sat_lo = max(_SATURATION_RANGE[0], saturation * (1 - span))
        sat_hi = min(_SATURATION_RANGE[1], saturation * (1 + span))
        for candidate_cap in np.linspace(cap_lo, cap_hi, 40):
            for candidate_sat in np.linspace(sat_lo, sat_hi, 40):
                value = loss(candidate_cap, candidate_sat)
                if value < best[0]:
                    best = (value, float(candidate_cap), float(candidate_sat))
        _, cap, saturation = best

    return round(cap, 3), round(saturation, 1)
