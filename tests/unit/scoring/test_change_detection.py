"""Noticing that a source's population changed, without being told.

Regime partitioning keys on the model and the prompt version, so it cannot see
a change that leaves both alone. The case already on the horizon: movie
enrichment adds a synopsis to `extraction_input`, films start earning more tags
with no prompt change at all, and every film re-extracts at once.
"""

from __future__ import annotations

from src.scoring.change_detection import DECISION_INTERVAL, MINIMUM_TO_DECLARE, detect_change
from src.scoring.curve_fit import Observation, expected_tags

GLOBAL = (3.7, 125.0)


def _rows(source: str, n: int, factor: float, start: int = 0) -> list[Observation]:
    """Rows earning `factor` times the global prediction, with spread.

    The noise is deterministic and it is not decoration: a detector
    standardises against the variance of normal behaviour, and a noise-free
    baseline has none — measured, the first version of this fixture produced
    sigma = 0 and every source was skipped. Real residuals always vary.
    """
    lengths = [40 + (i * 53) % 600 for i in range(n)]
    return [
        Observation(
            event_id=f"{source}-{start + i}",
            chars=c,
            tags=expected_tags(c, *GLOBAL) * factor + 0.25 * (((start + i) % 5) - 2),
            source=source,
        )
        for i, c in enumerate(lengths)
    ]


class TestDetection:
    def test_a_steady_source_reports_no_change(self):
        rows = _rows("steady", 120, 1.0)

        assert detect_change(rows, *GLOBAL) == {}

    def test_a_source_that_starts_earning_more_is_flagged(self):
        """The movie-enrichment shape: a step, not drift."""
        rows = _rows("films", 60, 1.0) + _rows("films", 60, 1.6, start=60)

        assert "films" in detect_change(rows, *GLOBAL)

    def test_a_source_that_starts_earning_fewer_is_flagged(self):
        rows = _rows("terse", 60, 1.0) + _rows("terse", 60, 0.5, start=60)

        assert "terse" in detect_change(rows, *GLOBAL)

    def test_the_change_point_is_near_where_it_happened(self):
        rows = _rows("films", 60, 1.0) + _rows("films", 60, 1.6, start=60)

        at = detect_change(rows, *GLOBAL)["films"]

        assert 55 <= at <= 95

    def test_two_odd_events_cannot_invent_a_regime(self):
        """A declaration floor: below it there is nothing to fit afterwards, so
        a change point would only strand the source below its arming threshold."""
        rows = _rows("noisy", 120, 1.0) + _rows("noisy", MINIMUM_TO_DECLARE - 1, 3.0, start=120)

        assert "noisy" not in detect_change(rows, *GLOBAL)

    def test_one_source_changing_does_not_flag_another(self):
        """Per-source, because a single global detector would reset everything
        every time one feed altered its listings."""
        rows = _rows("steady", 80, 1.0) + _rows("films", 60, 1.0) + _rows("films", 60, 1.6, start=60)

        flagged = detect_change(rows, *GLOBAL)

        assert "films" in flagged
        assert "steady" not in flagged

    def test_a_source_with_almost_no_rows_is_left_alone(self):
        assert detect_change(_rows("tiny", 3, 4.0), *GLOBAL) == {}

    def test_the_interval_is_the_conventional_one(self):
        assert DECISION_INTERVAL == 5.0
