"""What the refit is allowed to do once the gate has spoken.

Arming decides when to start, regimes decide what may be fitted together,
bounds refuse the impossible, and EWMA decides how fast. Four separate jobs;
conflating them is how a refit becomes inert or wild.
"""

from __future__ import annotations

import pytest

from src.scoring.curve_fit import Observation, expected_tags
from src.scoring.refit_policy import (
    ARMING_ROWS,
    CAP_BOUNDS,
    SATURATION_BOUNDS,
    RefitPolicy,
    plan_refit,
    smooth,
    within_bounds,
)

STATIC = (5.0, 190.0)


def _rows(n: int, cap: float = 3.7, saturation: float = 125.0, regime: str = "m/v1"):
    lengths = [30 + (i * 37) % 900 for i in range(n)]
    return [
        Observation(
            event_id=f"evt-{regime}-{i}",
            chars=c,
            tags=expected_tags(c, cap, saturation) + 0.3 * ((i % 7) - 3),
            source_type="s",
            regime=regime,
        )
        for i, c in enumerate(lengths)
    ]


class TestArming:
    def test_below_the_threshold_nothing_moves(self):
        outcome = plan_refit(_rows(ARMING_ROWS - 1), incumbent=STATIC)

        assert outcome.parameters == STATIC
        assert not outcome.decision.accepted
        assert "arm" in (outcome.decision.reason or "")

    def test_at_the_threshold_it_fits(self):
        outcome = plan_refit(_rows(ARMING_ROWS), incumbent=STATIC)

        assert outcome.decision.accepted

    def test_holding_is_silent_not_an_error(self):
        """A fresh deployment sits here for weeks. It is the normal state."""
        outcome = plan_refit(_rows(5), incumbent=STATIC)

        assert outcome.parameters == STATIC


class TestRegimes:
    def test_rows_from_two_regimes_are_never_blended(self):
        """A prompt change is a step, not drift. Fitting across one describes a
        population that never existed."""
        rows = _rows(ARMING_ROWS, regime="m/v1") + _rows(ARMING_ROWS, cap=8.0, regime="m/v2")

        outcome = plan_refit(rows, incumbent=STATIC, regime="m/v1")

        assert outcome.decision.train_rows + outcome.decision.holdout_rows == ARMING_ROWS

    def test_a_new_regime_starts_cold(self):
        """It holds the incumbent until it earns its own fit, rather than
        inheriting one drawn from a population it is not part of."""
        rows = _rows(ARMING_ROWS, regime="m/v1") + _rows(10, cap=8.0, regime="m/v2")

        outcome = plan_refit(rows, incumbent=STATIC, regime="m/v2")

        assert outcome.parameters == STATIC
        assert not outcome.decision.accepted

    def test_the_newest_regime_is_used_when_none_is_named(self):
        rows = _rows(ARMING_ROWS, regime="m/v1")

        assert plan_refit(rows, incumbent=STATIC).regime == "m/v1"


class TestBounds:
    def test_a_value_inside_the_domain_is_allowed(self):
        assert within_bounds(3.7, 124.0)

    def test_a_cap_below_one_tag_is_refused(self):
        assert not within_bounds(CAP_BOUNDS[0] - 0.1, 124.0)

    def test_an_absurd_saturation_is_refused(self):
        assert not within_bounds(3.7, SATURATION_BOUNDS[1] + 1)

    def test_the_bounds_are_wide_enough_for_a_changed_model(self):
        """A rail, not a rate limit. The fit must be free to move a long way if
        the data really says so."""
        assert within_bounds(1.5, 25.0)
        assert within_bounds(11.0, 1800.0)

    def test_an_out_of_bounds_fit_leaves_the_incumbent_alone(self):
        outcome = plan_refit(
            _rows(ARMING_ROWS, cap=40.0, saturation=5.0), incumbent=STATIC
        )

        assert outcome.parameters == STATIC


class TestSmoothing:
    def test_an_accepted_move_arrives_gradually(self):
        assert smooth(5.0, 3.7, alpha=0.15) == pytest.approx(4.805)

    def test_it_converges_on_the_target(self):
        value = 5.0
        for _ in range(60):
            value = smooth(value, 3.7, alpha=0.15)

        assert value == pytest.approx(3.7, abs=0.01)

    def test_the_planned_move_is_smoothed_not_jumped(self):
        outcome = plan_refit(_rows(ARMING_ROWS), incumbent=STATIC)

        assert outcome.decision.accepted
        assert outcome.parameters[0] < STATIC[0]
        assert outcome.parameters[0] > outcome.decision.candidate[0]

    def test_a_policy_may_move_at_a_different_pace(self):
        fast = plan_refit(_rows(ARMING_ROWS), incumbent=STATIC, policy=RefitPolicy(alpha=1.0))

        assert fast.parameters == fast.decision.candidate
