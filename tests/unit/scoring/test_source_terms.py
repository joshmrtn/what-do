"""Per-source multipliers on the global curve.

One curve cannot describe every source: measured, `cinema_veezi` earns ~⅓ more
tags per character than the global fit predicts, while `northshorenightout`
earns fewer. Separate curves are impossible — `do617` has two rows — so the
term is a multiplier, shrunk toward 1.0 by how much evidence there is for it.
"""

from __future__ import annotations

import pytest

from src.scoring.curve_fit import Observation, expected_tags
from src.scoring.source_terms import SHRINKAGE, source_multipliers

GLOBAL = (3.7, 125.0)


def _rows(source: str, n: int, factor: float) -> list[Observation]:
    """Rows earning `factor` times what the global curve predicts."""
    lengths = [40 + (i * 53) % 600 for i in range(n)]
    return [
        Observation(
            event_id=f"{source}-{i}",
            chars=c,
            tags=expected_tags(c, *GLOBAL) * factor,
            source_type=source,
        )
        for i, c in enumerate(lengths)
    ]


class TestSourceMultipliers:
    def test_a_source_matching_the_global_curve_gets_one(self):
        k = source_multipliers(_rows("plain", 80, 1.0), *GLOBAL)

        assert k["plain"] == pytest.approx(1.0, abs=0.02)

    def test_a_source_earning_more_tags_gets_more_than_one(self):
        k = source_multipliers(_rows("rich", 80, 1.4), *GLOBAL)

        assert k["rich"] > 1.2

    def test_a_source_earning_fewer_gets_less_than_one(self):
        k = source_multipliers(_rows("terse", 80, 0.7), *GLOBAL)

        assert k["terse"] < 0.85

    def test_a_source_with_barely_any_rows_is_pulled_toward_the_global(self):
        """`do617` had two rows and a raw ratio of 1.40. Taking that literally
        would let two events rewrite a source's expectations."""
        k = source_multipliers(_rows("tiny", 2, 1.4), *GLOBAL)

        assert k["tiny"] < 1.2

    def test_evidence_earns_influence(self):
        """The same raw ratio moves further when more rows support it."""
        few = source_multipliers(_rows("s", 3, 1.4), *GLOBAL)["s"]
        many = source_multipliers(_rows("s", 200, 1.4), *GLOBAL)["s"]

        assert many > few

    def test_an_unseen_source_is_absent_so_a_caller_defaults_to_one(self):
        """Cold start, solved by having nothing to say: a brand-new source uses
        the global curve until it has earned an opinion."""
        k = source_multipliers(_rows("known", 40, 1.0), *GLOBAL)

        assert "brand-new" not in k
        assert k.get("brand-new", 1.0) == 1.0

    def test_it_is_recomputed_from_whatever_it_is_given(self):
        """Nightly, never frozen: today's ratios are today's."""
        before = source_multipliers(_rows("s", 60, 1.0), *GLOBAL)["s"]
        after = source_multipliers(_rows("s", 60, 1.5), *GLOBAL)["s"]

        assert after > before

    def test_no_rows_yields_no_terms(self):
        assert source_multipliers([], *GLOBAL) == {}

    def test_shrinkage_is_the_measured_value(self):
        """n₀=5 beat both no shrinkage and heavier shrinkage on held-out folds,
        at +0.05 R² over the global curve alone."""
        assert SHRINKAGE == 5
