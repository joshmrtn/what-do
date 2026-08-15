"""Detecting that a source's population changed, without being told.

Regime partitioning keys on `extraction_model` and `extraction_prompt_version`,
so it cannot see a change that leaves both alone. The case already on the
horizon: **movie enrichment adds a synopsis to `extraction_input`**, so films
start earning more tags with no prompt change at all — and every film
re-extracts at once, making it a step rather than drift.

Per-source, deliberately. A single global detector would reset everything every
time one feed altered its listings.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from src.scoring.curve_fit import Observation

#: Allowance, in standard deviations. Residuals smaller than this are noise and
#: accumulate nothing, which is what stops ordinary variation drifting the sum.
ALLOWANCE = 0.5

#: Decision interval, in standard deviations. The conventional 5σ: slow on
#: noise, quick on a genuine step.
DECISION_INTERVAL = 5.0

#: Rows a source needs after a suspected change before it may be *declared*.
#: Below this there is nothing to fit afterwards, so declaring would only strand
#: the source below its arming threshold on the strength of a couple of events.
MINIMUM_TO_DECLARE = 10

#: Rows a source needs at all before it is watched.
MINIMUM_TO_WATCH = 30


def detect_change(
    rows: list[Observation], cap: float, saturation: float
) -> dict[str, int]:
    """Where each source's behaviour changed, if it did.

    Two-sided CUSUM over standardised residuals, in the order the rows are
    given — which the caller should make chronological.

    Args:
        rows: Observations across sources, all from one regime.
        cap: The global curve's cap.
        saturation: The global curve's saturation.

    Returns:
        `source_type -> index of the change point`, for sources that changed.
        An absent source did not, and most nights the result is empty.

    A firing is not a reason to freeze. It declares a new regime for that
    source, whose pre-change rows are then dropped — so the arming threshold
    applies again and the last accepted curve holds until the new regime earns
    its own. That is the arming rule doing its ordinary job, not a policy of
    standing still because something moved.
    """
    by_source: dict[str, list[Observation]] = defaultdict(list)
    for row in rows:
        by_source[row.source_type].append(row)

    changes: dict[str, int] = {}
    for source, source_rows in by_source.items():
        if len(source_rows) < MINIMUM_TO_WATCH:
            continue

        chars = np.array([r.chars for r in source_rows], dtype=float)
        tags = np.array([r.tags for r in source_rows], dtype=float)
        residual = tags - cap * (1.0 - np.exp(-chars / saturation))

        # Standardised against an **in-control baseline**, not the whole
        # series. Centring on the overall mean folds the change itself into the
        # reference, so every pre-change row reads as shifted and the sum
        # crosses almost immediately — measured, a step at row 60 was reported
        # at row 11.
        baseline = residual[:MINIMUM_TO_WATCH]
        sigma = float(np.std(baseline))
        if sigma <= 0:
            continue
        standardised = (residual - float(np.mean(baseline))) / sigma

        high = low = 0.0
        for index, value in enumerate(standardised):
            if index < MINIMUM_TO_WATCH:
                # The baseline window defines "normal"; it cannot also be
                # evidence against it.
                continue
            high = max(0.0, high + value - ALLOWANCE)
            low = min(0.0, low + value + ALLOWANCE)
            if max(high, -low) <= DECISION_INTERVAL:
                continue
            # Enough must remain after the change for the source to be fittable
            # again; otherwise declaring only strands it below its threshold.
            if len(source_rows) - index < MINIMUM_TO_DECLARE:
                break
            changes[source] = index
            break

    return changes
