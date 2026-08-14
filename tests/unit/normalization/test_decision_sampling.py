"""Which dedup decisions are kept, and at what rate.

Keeping every comparison is affordable per night (~1,631) but not per year.
Keeping only the interesting ones throws away the majority class entirely, and
a model validated without it has never seen what it will mostly encounter.

So: everything that merged, everything close to merging, and a deterministic
slice of the rest — with the rate recorded, because a model reweighting to true
prevalence needs the rate that produced *that* row.
"""

from __future__ import annotations

from src.normalization.decision_sampling import (
    STRATUM_MERGED,
    STRATUM_NEAR_MISS,
    STRATUM_SAMPLED,
    select_for_storage,
)
from src.normalization.deduplicator import MergeDecision


def _decision(score: float, verdict: str = "distinct", a: str = "a", b: str = "b") -> MergeDecision:
    return MergeDecision(
        pass_name="semantic",
        record_kind="event",
        record_a=a,
        record_b=b,
        score=score,
        verdict=verdict,
        content_hash_a=f"hash-of-{a}",
        content_hash_b=f"hash-of-{b}",
    )


def _kept(decisions, *, floor=0.70, denominator=10):
    return select_for_storage(decisions, floor=floor, denominator=denominator)


class TestEveryPositiveIsKept:
    def test_a_merge_is_always_stored(self):
        """The rarest class. Two in the whole live corpus — losing one to a
        sampling rule would be losing half the positives."""
        kept = _kept([_decision(0.95, verdict="merged")])

        assert len(kept) == 1
        assert kept[0].stratum == STRATUM_MERGED

    def test_a_merge_below_the_floor_is_still_stored(self):
        """A pass may merge on evidence other than this score. Whatever the
        score, a merge is a positive and positives are never sampled away."""
        kept = _kept([_decision(0.10, verdict="merged")])

        assert len(kept) == 1
        assert kept[0].stratum == STRATUM_MERGED


class TestNearMissesAreKept:
    def test_a_rejection_above_the_floor_is_stored(self):
        """0.90 against a 0.92 threshold: the hard case, and what the dataset
        is for."""
        kept = _kept([_decision(0.90)])

        assert len(kept) == 1
        assert kept[0].stratum == STRATUM_NEAR_MISS

    def test_the_floor_itself_counts_as_near(self):
        kept = _kept([_decision(0.70)])

        assert kept[0].stratum == STRATUM_NEAR_MISS


class TestEasyNegativesAreDownsampled:
    def _below_floor(self, n: int) -> list[MergeDecision]:
        return [_decision(0.45, a=f"e{i}", b="z") for i in range(n)]

    def test_most_are_dropped(self):
        kept = _kept(self._below_floor(200), denominator=10)

        assert 5 < len(kept) < 60, "roughly a tenth, not all and not none"
        assert all(k.stratum == STRATUM_SAMPLED for k in kept)

    def test_the_same_pair_is_always_decided_the_same_way(self):
        """Determinism is the property. A pair that flickers in and out night
        to night is worse than one consistently dropped: the dataset would
        churn and the same comparison would appear and vanish for no reason."""
        decisions = self._below_floor(200)

        first = {(k.decision.record_a, k.decision.record_b) for k in _kept(decisions)}
        second = {(k.decision.record_a, k.decision.record_b) for k in _kept(decisions)}

        assert first == second

    def test_the_order_it_arrives_in_does_not_change_what_is_kept(self):
        """Sampling on position rather than identity would keep a different
        slice every night, for no reason a reader could ever recover."""
        decisions = self._below_floor(200)

        forward = {(k.decision.record_a, k.decision.record_b) for k in _kept(decisions)}
        backward = {
            (k.decision.record_a, k.decision.record_b)
            for k in _kept(list(reversed(decisions)))
        }

        assert forward == backward

    def test_a_denominator_of_one_keeps_everything(self):
        """The escape hatch: sampling off, for a run that wants the lot."""
        kept = _kept(self._below_floor(50), denominator=1)

        assert len(kept) == 50


class TestTheRateIsRecordedOnTheRow:
    def test_a_sampled_row_carries_the_denominator_that_produced_it(self):
        """Reweighting to true prevalence needs the rate that produced *this*
        row, and the rate will eventually be tuned. A rule that lives only in
        config becomes unrecoverable the moment it moves."""
        kept = _kept(self._many(), denominator=10)

        assert all(k.sample_denominator == 10 for k in kept if k.stratum == STRATUM_SAMPLED)

    def test_a_fully_kept_stratum_records_a_denominator_of_one(self):
        """Not the run's sampling rate: these rows were not sampled at all, and
        recording 10 here would tell a model they were one in ten of their kind."""
        kept = _kept([_decision(0.95, verdict="merged"), _decision(0.90)], denominator=10)

        assert {k.sample_denominator for k in kept} == {1}

    @staticmethod
    def _many() -> list[MergeDecision]:
        return [_decision(0.45, a=f"e{i}", b="z") for i in range(200)]
