"""Per-source multipliers on the global tag-confidence curve.

One curve cannot describe every source. Measured over 273 real rows,
`cinema_veezi` earns about a third more tags per character than the global fit
predicts — film listings are terse but tag-dense — while `northshorenightout`
and `cinema_capeann` earn slightly fewer.

Separate curves per source are impossible: `do617` has two rows. So the term is
a multiplier on the shared curve, shrunk toward 1.0 by how much evidence there
is for it, which is what makes a two-row source safe.
"""

from __future__ import annotations

import math
from collections import defaultdict

from src.scoring.curve_fit import Observation

#: Shrinkage constant. A source's own ratio carries weight `n / (n + SHRINKAGE)`
#: and the rest is pulled to 1.0, so influence is earned by evidence.
#:
#: Measured 2026-08-14 on five-fold cross-validation: adding this term improved
#: held-out R² from 0.663 to 0.713, and 5 beat both no shrinkage and heavier
#: shrinkage. It generalises rather than fitting noise.
SHRINKAGE = 5


def source_multipliers(
    rows: list[Observation], cap: float, saturation: float
) -> dict[str, float]:
    """How much more or less each source earns than the global curve says.

    Recomputed from whatever it is given, every night — a frozen ratio is a
    measurement of a population that has moved on.

    Args:
        rows: Observations, all from one regime.
        cap: The global curve's cap.
        saturation: The global curve's saturation.

    Returns:
        `source_type -> multiplier`. A source that is absent has said nothing,
        and a caller should read that as 1.0 — which is cold start solved by
        having no opinion rather than by a special case.
    """
    actual: dict[str, float] = defaultdict(float)
    predicted: dict[str, float] = defaultdict(float)
    counts: dict[str, int] = defaultdict(int)

    for row in rows:
        actual[row.source_type] += row.tags
        predicted[row.source_type] += cap * (1.0 - math.exp(-row.chars / saturation))
        counts[row.source_type] += 1

    terms: dict[str, float] = {}
    for source, total in predicted.items():
        if total <= 0:
            continue
        raw = actual[source] / total
        n = counts[source]
        # Shrunk toward 1.0: a source with two rows barely moves, one with
        # hundreds moves nearly all the way. Taking `raw` literally would let
        # two events rewrite a source's expectations.
        terms[source] = 1.0 + (n / (n + SHRINKAGE)) * (raw - 1.0)
    return terms
