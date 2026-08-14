"""Which dedup decisions are worth storing, and at what rate.

Every comparison a pass makes is reported, but not every one is kept. The live
corpus scores ~1,631 pairs a night, of which two merge: affordable for a night,
not for a year, and overwhelmingly the same easy negative over and over.

Dropping the easy negatives entirely would be worse than keeping too many. They
are the class a dedup model will mostly meet in production, and a model
validated without them has never seen its own working conditions. So they are
downsampled rather than discarded, and the rate travels with the row.

The full pair population stays regenerable regardless: nothing is deleted, so
recomputing the guards and the cosines over stored records reproduces every
comparison. What cannot be recomputed — the verdict under the thresholds in
force at the time — is what gets stored.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from src.normalization.deduplicator import VERDICT_MERGED, MergeDecision

#: The pair merged. The rarest class, never sampled away.
STRATUM_MERGED = "merged"
#: Compared, rejected, and close enough that the rejection is informative.
STRATUM_NEAR_MISS = "near_miss"
#: Compared, rejected, and unremarkable. Downsampled.
STRATUM_SAMPLED = "sampled"

#: What a row records when its stratum was kept whole. Not the run's sampling
#: rate: these rows were never sampled, and recording the rate here would tell
#: a reader they were one in N of their kind.
_KEPT_IN_FULL = 1


@dataclass(frozen=True)
class SampledDecision:
    """A decision selected for storage, with why it survived selection."""

    decision: MergeDecision
    stratum: str
    sample_denominator: int


def _stratum(decision: MergeDecision, floor: float) -> str:
    if decision.verdict == VERDICT_MERGED:
        return STRATUM_MERGED
    return STRATUM_NEAR_MISS if decision.score >= floor else STRATUM_SAMPLED


def _in_sample(decision: MergeDecision, denominator: int) -> bool:
    """Whether this pair is one of the kept fraction.

    Keyed on the pair's identity, never on its position or a random draw, so
    the same pair is always kept or always dropped. A pair that flickered in
    and out night to night would churn the dataset and leave a reader unable to
    recover why a comparison they saw last week is gone.
    """
    if denominator <= 1:
        return True
    key = f"{decision.pass_name}|{decision.record_a}|{decision.record_b}"
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % denominator == 0


def select_for_storage(
    decisions: list[MergeDecision], *, floor: float, denominator: int
) -> list[SampledDecision]:
    """Choose which decisions to keep, and record the rate that chose them.

    Args:
        decisions: Every comparison a pass made this run.
        floor: Score at or above which a *rejection* is kept in full. Sits above
            the embedding noise floor, so it catches the tail without drowning
            in the mode.
        denominator: Keep one in this many of the rest. 1 keeps everything.

    Returns:
        The kept decisions, each carrying its stratum and the denominator that
        applied to it.
    """
    kept: list[SampledDecision] = []
    for decision in decisions:
        stratum = _stratum(decision, floor)
        if stratum == STRATUM_SAMPLED:
            if not _in_sample(decision, denominator):
                continue
            kept.append(SampledDecision(decision, stratum, denominator))
        else:
            kept.append(SampledDecision(decision, stratum, _KEPT_IN_FULL))
    return kept
