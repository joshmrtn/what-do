"""Deciding whether a refit is allowed to move the curve.

A gate refuses a bad move; a rate limit only slows one down. This is the
safety mechanism, so it is the one that must not be clever.
"""

from __future__ import annotations

import pytest

from src.scoring.curve_fit import Observation, expected_tags
from src.scoring.refit_gate import assign_fold, consider_refit

INCUMBENT = (5.0, 190.0)


def _expected_fold(event_id: str, folds: int) -> int:
    """The fold an id must land in, computed independently of the code."""
    import hashlib

    digest = hashlib.sha256(event_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % folds


def _rows(cap: float, saturation: float, n: int = 60, noise: float = 0.0) -> list[Observation]:
    """Observations drawn from a known curve, with realistic lengths.

    `noise` is deterministic rather than random — the point of hash-assigned
    folds is reproducibility, and a fixture that needed a seed would undercut
    the thing being tested. Without it every subset fits the generating curve
    exactly, so a fit on four fifths and a fit on five are indistinguishable and
    a test asserting which one is adopted proves nothing.
    """
    lengths = [30 + (i * 37) % 900 for i in range(n)]
    return [
        Observation(
            event_id=f"evt-{i}",
            chars=c,
            tags=expected_tags(c, cap, saturation) + noise * ((i % 7) - 3),
            source="s",
        )
        for i, c in enumerate(lengths)
    ]


class TestFolds:
    def test_a_fold_is_stable_for_the_same_id(self):
        """Hash-assigned, never RNG: the same corpus must refit identically on
        two runs, or a past score stops being recomputable from stored data."""
        assert assign_fold("evt-1", 5) == assign_fold("evt-1", 5)

    def test_a_fold_does_not_depend_on_the_corpus_around_it(self):
        """An id keeps its fold whether it arrives in a corpus of 50 or 5,000,
        so tonight's events cannot reshuffle yesterday's split and silently
        change the verdict. Asserted against a computed digest rather than
        against another call, which would hold for any pure function."""
        assert assign_fold("evt-1", 5) == _expected_fold("evt-1", 5)
        assert assign_fold("a-much-later-event", 5) == _expected_fold(
            "a-much-later-event", 5
        )

    def test_it_does_not_use_the_interpreter_hash(self):
        """`hash()` is salted per process, so folds drawn from it would differ
        between two runs of the same batch."""
        assert assign_fold("evt-1", 5) == _expected_fold("evt-1", 5)

    def test_folds_are_spread_across_the_range(self):
        seen = {assign_fold(f"evt-{i}", 5) for i in range(500)}

        assert seen == {0, 1, 2, 3, 4}

    def test_the_spread_is_roughly_even(self):
        counts = [0] * 5
        for i in range(1000):
            counts[assign_fold(f"evt-{i}", 5)] += 1

        assert min(counts) > 120, counts


class TestGate:
    def test_a_better_candidate_is_accepted(self):
        """The corpus says 3.7/125 and the incumbent says 5.0/190."""
        decision = consider_refit(_rows(3.7, 125.0), incumbent=INCUMBENT)

        assert decision.accepted
        assert decision.candidate[0] == pytest.approx(3.7, abs=0.3)

    def test_a_candidate_that_fits_worse_is_refused(self):
        """The incumbent already describes this corpus, so nothing should move."""
        decision = consider_refit(_rows(*INCUMBENT), incumbent=INCUMBENT)

        assert not decision.accepted
        assert decision.candidate is None

    def test_the_accepted_fit_comes_from_the_whole_corpus(self):
        """Fit on 80% to *judge*, then refit on 100% to *use* — the held-out
        fifth is spent on the decision, not thrown away from the answer.

        Needs noisy rows: on a clean curve every subset fits identically, so
        adopting the trial fit and adopting the full fit are the same value and
        the assertion holds either way.
        """
        from src.scoring.curve_fit import fit_curve

        rows = _rows(3.7, 125.0, noise=0.35)
        train = [r for r in rows if assign_fold(r.event_id) != 0]

        decision = consider_refit(rows, incumbent=INCUMBENT)

        assert decision.candidate == fit_curve(rows)
        assert decision.candidate != fit_curve(train), "fixture is not discriminating"

    def test_both_sides_are_scored_on_the_same_held_out_rows(self):
        decision = consider_refit(_rows(3.7, 125.0), incumbent=INCUMBENT)

        assert decision.holdout_rows > 0
        assert decision.incumbent_score is not None
        assert decision.candidate_score is not None
        assert decision.candidate_score < decision.incumbent_score  # lower loss wins

    def test_the_decision_reports_what_it_saw(self):
        """Phase 7 writes this to `run_history`: without it a past score cannot
        be explained even with the constants in hand."""
        decision = consider_refit(_rows(3.7, 125.0), incumbent=INCUMBENT)

        assert decision.train_rows + decision.holdout_rows == 60

    def test_too_few_rows_to_hold_any_out_is_refused(self):
        decision = consider_refit(_rows(3.7, 125.0, n=12), incumbent=INCUMBENT)

        assert not decision.accepted
        assert "rows" in (decision.reason or "")

    def test_the_same_corpus_decides_the_same_way_twice(self):
        rows = _rows(3.7, 125.0)

        assert consider_refit(rows, incumbent=INCUMBENT) == consider_refit(
            rows, incumbent=INCUMBENT
        )
