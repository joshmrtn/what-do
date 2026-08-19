"""The one-way churn latch: acting on the measurement rather than logging it.

The churn detector has reported into a run summary since it shipped, and a log
line only works if somebody reads it. This arms the measurement: once a
publisher's identifiers have been shown decisively not to identify anything, the
source moves onto content-derived ids and stays there.

**One way, permanently.** Not only because a publisher that has done this is not
to be trusted again, but because a content-keyed source reads 0% churn *by
construction* — anything re-evaluating in both directions would oscillate,
re-keying and minting duplicates each way.

**Evidence accumulates; it is never a streak.** Measured 2026-08-17:
`northshorenightout` held its UIDs for a whole day, so the detector reported a
clean 0% for a feed that re-mints everything. A streak counter resets on that,
and a source churning every *other* night would never reach two consecutive
qualifying runs — accumulating duplicates for ever while the latch waited. That
is the failure this exists to prevent, reintroduced by the guard meant to make
it safe.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from src.config import IDENTITY_CONTENT, IDENTITY_PUBLISHER, SourcesConfig
from src.ingestion.id_churn import ChurnTally
from src.ingestion.identity import ContentIdRule
from src.storage.identity_state import IdentityStateStore

#: Churned listings, summed across qualifying runs, before the latch may fire.
#: Deliberately a **total** and not a per-run floor: a per-run minimum does not
#: make a small feed slower to latch, it makes it impossible. Three live feeds
#: never see more than five listings in a night.
MIN_EVIDENCE = 20

#: How many separate runs must have contributed. Guards against one anomalous
#: fetch permanently changing how a source is keyed.
MIN_RUNS = 2


@dataclass
class LatchReport:
    """What the latch did this run, for the batch summary."""

    #: Sources moved onto content ids by this run. Rare and worth announcing.
    latched: list[str] = field(default_factory=list)
    #: Pinned to `publisher`, and churning anyway. The pin suppresses the
    #: action, never the observation, so a wrong pin is visible rather than
    #: silent.
    pinned_but_churning: list[str] = field(default_factory=list)


def arm_latches(
    churn: dict[str, ChurnTally],
    *,
    state: IdentityStateStore,
    sources: SourcesConfig,
    at: datetime,
) -> LatchReport:
    """Record this run's evidence and latch any source that has earned it.

    Args:
        churn: This run's per-source tallies, keyed on `source` — the feed,
            never `source_type`, which is a category covering feeds with
            opposite behaviour.
        state: Where the accumulated evidence lives.
        sources: The configured policy, which decides whether a latch may act.
        at: Injected clock.

    Returns:
        What fired, and which pinned sources are churning anyway.
    """
    report = LatchReport()

    for source, tally in sorted(churn.items()):
        state.record(source, tally, at=at)
        current = state.get(source)
        if not _has_earned_it(current.churn_evidence, current.qualifying_runs):
            continue

        policy = sources.identity_for(source)
        if policy == IDENTITY_PUBLISHER:
            report.pinned_but_churning.append(source)
            continue
        if policy == IDENTITY_CONTENT or current.latched_at is not None:
            continue

        state.latch(source, at=at)
        report.latched.append(source)

    return report


def latched_rule(sources: SourcesConfig, *, state: IdentityStateStore) -> ContentIdRule:
    """The rule the adapters read: config, plus everything already latched.

    The latched set is read **once**, not per candidate. Every candidate in a
    run asks this question and the answer cannot change mid-run, so a query
    apiece would be thousands of round trips for one fact.
    """
    latched = state.latched()

    def rule(source: str) -> bool:
        if sources.identity_for(source) == IDENTITY_CONTENT:
            return True
        return source in latched

    return rule


def _has_earned_it(evidence: int, runs: int) -> bool:
    return evidence >= MIN_EVIDENCE and runs >= MIN_RUNS
