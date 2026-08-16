"""Measure whether a source's own identifiers actually identify anything.

The raw layer rests on ids being stable across fetches — `last_seen_at`,
`candidate_versions` and the edit-rate measurement all assume it, and **nothing
checked it**. Measured 2026-08-15, `northshorenightout` re-mints every Google
Calendar UID between days: 138 events, 138 ids, none of them seen before, every
night. The duplicates were only noticed because a dedup row count looked odd.

Id stability cannot be known about a source in advance; it can only be observed.
So the batch observes it, per source, every run.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.models.event_candidate import EventCandidate
from src.normalization.deduplicator import canonical_title, canonical_venue

#: A listing's identity independent of whatever id the publisher attached to it.
ContentKey = tuple[str, str, str, str]


@dataclass(frozen=True)
class ChurnTally:
    """How far one source's identifiers can be trusted, this run."""

    #: Fetched candidates whose listing was already stored. Only these can say
    #: anything about id stability.
    seen_before: int = 0
    #: Of those, the ones that arrived under an id we have never seen.
    churned: int = 0

    @property
    def rate(self) -> float | None:
        """The share of already-known listings that arrived with a new id.

        `None` where nothing was seen before — a source's first run stores
        everything and matches nothing. Zero would read as "ids are stable",
        which is the opposite of what is known at that point.
        """
        if self.seen_before == 0:
            return None
        return self.churned / self.seen_before


def content_key(candidate: EventCandidate) -> ContentKey:
    """What identifies the listing itself, rather than the publisher's handle on it.

    Scoped to the **feed**, not the category: a calendar feed and a listing page
    legitimately cover the same event, so a key spanning `source_type` would let
    one feed's stored row make the other's first publication look already known,
    and its correct, stable, genuinely new id would be counted as churn.

    Canonical on both text fields, because a re-cased republish would otherwise
    read as a brand new listing and hide the very churn this measures — the same
    reason dedup compares on a canonical key rather than raw strings.

    Carries the **start**: every occurrence of a recurring programme shares a
    title and venue, so a key without it collapses a whole season into one
    listing and reports the season as churn.
    """
    return (
        candidate.source,
        canonical_title(candidate.title or ""),
        canonical_venue(candidate.venue or ""),
        candidate.start_time.isoformat() if candidate.start_time else "",
    )


def churn_by_source(
    fetched: list[EventCandidate], *, stored: list[EventCandidate]
) -> dict[str, ChurnTally]:
    """Per feed, how many already-known listings arrived under a new id.

    Keyed on `source` — the feed — rather than `source_type`, which is a
    *category* and can hold several feeds with opposite identity policies
    (#34). Live, `source_type = 'northshorenightout'` covers an ICS feed that
    re-mints every Google UID and a listing page keyed on a content hash;
    averaged, 100% and 0% reported 60% and described neither.

    Genuinely new listings are **outside the rate entirely**. They carry no
    evidence either way, and counting them would dilute the measurement with
    every real addition a healthy feed publishes.

    Args:
        fetched: This run's candidates, as fetched.
        stored: Candidates already persisted.

    Returns:
        One tally per feed (`source`) present in `fetched`.
    """
    known_ids = {candidate.id for candidate in stored}
    known_keys = {content_key(candidate) for candidate in stored}

    seen_before: dict[str, int] = {}
    churned: dict[str, int] = {}

    for candidate in fetched:
        source = candidate.source
        seen_before.setdefault(source, 0)
        churned.setdefault(source, 0)

        if content_key(candidate) not in known_keys:
            continue

        seen_before[source] += 1
        if candidate.id not in known_ids:
            churned[source] += 1

    return {
        source: ChurnTally(seen_before=count, churned=churned[source])
        for source, count in seen_before.items()
    }
