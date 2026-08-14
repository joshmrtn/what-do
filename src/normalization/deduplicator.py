"""Deduplication engine — Pass 1 (fuzzy, pre-embedding)."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field, replace
import copy
from datetime import datetime, timedelta
from typing import Callable

from rapidfuzz import fuzz

from src.config import DeduplicationConfig
from src.models.event import Event

#: A pair the pass judged the same event.
VERDICT_MERGED = "merged"
#: A pair the pass compared and judged different. Most of the training signal.
VERDICT_DISTINCT = "distinct"


@dataclass(frozen=True)
class Comparison:
    """What a pass concluded about one pair it was able to compare.

    Separate from the verdict because the threshold belongs to the pass: the
    caller decides what counts as a duplicate, and this records what was
    measured either way.
    """

    score: float
    duplicate: bool


@dataclass(frozen=True)
class MergeDecision:
    """One comparison, as it will be stored and later read back by a person.

    `record_a` is always the lexicographically smaller id, so a pair has one
    identity regardless of the order it was iterated in.
    """

    pass_name: str
    record_kind: str
    record_a: str
    record_b: str
    score: float
    verdict: str
    #: Digests of the text each side was compared on, in `record_a`/`record_b`
    #: order. `event_candidates` is written with INSERT OR REPLACE (#27), so a
    #: re-fetched listing overwrites the row a reader would later check this
    #: decision against. These do not recover the original text; they make the
    #: substitution detectable instead of silent.
    content_hash_a: str
    content_hash_b: str


@dataclass(frozen=True)
class DedupResult:
    """The merged events, every decision that produced them, and the losers.

    A cluster is a labelled training scenario, and a destroyed loser cannot be
    one — so the events a merge absorbed come back marked rather than dropped.
    Callers persist them; the repositories filter them out of ordinary reads.
    """

    events: list[Event]
    decisions: list[MergeDecision]
    superseded: list[Event] = field(default_factory=list)


def content_fingerprint(text: str) -> str:
    """Stable digest of the text a comparison rested on."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _title_similarity(a: str | None, b: str | None) -> float | None:
    """How alike two titles are, or None when there is nothing to compare.

    Two absent titles score 1.0 rather than None: `_title_match` has always
    treated them as matching, and reporting no decision for a pair that *does*
    merge would leave the merge unexplained — the thing this exists to prevent.
    """
    if a is None and b is None:
        return 1.0
    if a is None or b is None:
        return None
    return float(fuzz.token_sort_ratio(a, b)) / 100.0


def venues_match(a: str | None, b: str | None) -> bool:
    """True when venues are considered the same (exact canonical match).

    Shared with Pass 2, which applies the same structural guard.
    """
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    return a == b


def times_match(
    a_time: datetime | None, b_time: datetime | None, window_hours: float
) -> bool:
    """True when start times are within the configured window.

    Shared with Pass 2, which applies the same structural guard.
    """
    if a_time is None and b_time is None:
        return True
    if a_time is None or b_time is None:
        return False
    return abs((a_time - b_time).total_seconds()) <= window_hours * 3600


def _listing_text(event: Event) -> str:
    """The listing as normalization received it — what a person would read."""
    return f"{event.title or ''}\n{event.description or ''}"


def _compare(a: Event, b: Event, cfg: DeduplicationConfig) -> Comparison | None:
    """Pass 1's comparison: the title, gated by the structural guards.

    The guards come first because they decide whether the pair is compared at
    all — a different venue is not a low score, it is no comparison. They also
    discard 99.87% of pairs, which is what makes recording the rest affordable.
    """
    if not venues_match(a.venue, b.venue):
        return None
    if not times_match(a.start_time, b.start_time, cfg.time_window_hours):
        return None

    similarity = _title_similarity(a.title, b.title)
    if similarity is None:
        return None
    return Comparison(score=similarity, duplicate=similarity >= cfg.fuzzy_title_threshold)


def _candidate_identity(event: Event) -> str | None:
    """The stable id for a Pass 1 record.

    Pass 1 runs before anything is stored, on events whose uuids are minted per
    run — keyed on those, every pair would be new every night and nothing would
    ever accumulate. At this point each event carries exactly one candidate,
    and candidate ids are deterministic.
    """
    return event.source_event_candidates[0] if event.source_event_candidates else None


def _null_count(event: Event) -> int:
    """Count None-valued optional fields — lower means more complete."""
    fields = [
        event.url, event.image_url, event.title, event.venue,
        event.description, event.location, event.start_time, event.end_time,
        event.summary, event.summary_embedding, event.weather, event.astronomical_data,
    ]
    return sum(1 for f in fields if f is None)


def merge_cluster(events: list[Event]) -> Event:
    """Merge a cluster of duplicate events into one canonical record.

    Most-complete record (fewest None fields) is the base. Tiebreak: earliest
    created_at. Non-None fields from secondary records fill gaps in the base.
    Source candidate IDs are unioned across all contributors.
    """
    ranked = sorted(events, key=lambda e: (_null_count(e), e.created_at))
    base = ranked[0]

    merged_candidates: list[str] = []
    for e in events:
        for cid in e.source_event_candidates:
            if cid not in merged_candidates:
                merged_candidates.append(cid)

    url = base.url
    image_url = base.image_url
    title = base.title
    venue = base.venue
    description = base.description
    location = base.location
    start_time = base.start_time
    end_time = base.end_time
    summary = base.summary
    summary_embedding = base.summary_embedding
    weather = base.weather
    astronomical_data = base.astronomical_data

    for secondary in ranked[1:]:
        if url is None:
            url = secondary.url
        if image_url is None:
            image_url = secondary.image_url
        if title is None:
            title = secondary.title
        if venue is None:
            venue = secondary.venue
        if description is None:
            description = secondary.description
        if location is None:
            location = secondary.location
        if start_time is None:
            start_time = secondary.start_time
        if end_time is None:
            end_time = secondary.end_time
        if summary is None:
            summary = secondary.summary
        if summary_embedding is None:
            summary_embedding = secondary.summary_embedding
        if weather is None:
            weather = secondary.weather
        if astronomical_data is None:
            astronomical_data = secondary.astronomical_data

    return replace(
        base,
        source_event_candidates=merged_candidates,
        url=url,
        image_url=image_url,
        title=title,
        venue=venue,
        description=description,
        location=location,
        start_time=start_time,
        end_time=end_time,
        summary=summary,
        summary_embedding=summary_embedding,
        weather=weather,
        astronomical_data=astronomical_data,
    )


def cluster_and_merge(
    events: list[Event],
    compare: Callable[[Event, Event], Comparison | None],
    identify: Callable[[Event], str | None],
    content_of: Callable[[Event], str],
    pass_name: str,
    record_kind: str,
    now: datetime | None = None,
) -> DedupResult:
    """Group events into clusters by a pairwise comparison and merge each cluster.

    Greedy union-find: duplicates are transitive, so if A matches B and B
    matches C, all three collapse into one record. Shared by both dedup passes,
    which differ only in how they compare.

    Every comparison the passes *make* is reported, not only the ones that
    merge. A rejection is the label a future dedup model most needs and the one
    a surviving row can never express: measured on the live corpus, two merges
    against 1,629 considered-and-rejected pairs.

    Args:
        events: Events to cluster.
        compare: Pairwise comparison, or None when the pair was never compared —
            a failed structural guard or a missing vector. None is *unknown*,
            not "different", so it records nothing.
        content_of: The text the comparison rested on, fingerprinted onto the
            decision so a later reader can tell whether the record has since
            been edited underneath it.
        identify: The stable id of the record a decision should name, or None
            when it has none. A decision keyed on an id that will not exist
            tomorrow would never match again, so it is not recorded at all.
        pass_name: Which pass is comparing — `fuzzy`, `semantic`, `reconcile`.
        record_kind: What `identify` returns — `candidate` or `event`.

    Returns:
        One merged Event per cluster (order not guaranteed), and every decision
        made along the way.
    """
    if not events:
        return DedupResult(events=[], decisions=[])

    n = len(events)
    parent = list(range(n))
    decisions: list[MergeDecision] = []

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        px, py = find(x), find(y)
        if px != py:
            parent[py] = px

    for i in range(n):
        for j in range(i + 1, n):
            comparison = compare(events[i], events[j])
            if comparison is None:
                continue
            if comparison.duplicate:
                union(i, j)

            left, right = identify(events[i]), identify(events[j])
            if left is None or right is None:
                continue
            # Sorted, so a pair has one identity however it is iterated and one
            # row wherever it is stored.
            fingerprints = {
                left: content_fingerprint(content_of(events[i])),
                right: content_fingerprint(content_of(events[j])),
            }
            record_a, record_b = sorted((left, right))
            decisions.append(
                MergeDecision(
                    pass_name=pass_name,
                    record_kind=record_kind,
                    record_a=record_a,
                    record_b=record_b,
                    score=comparison.score,
                    verdict=VERDICT_MERGED if comparison.duplicate else VERDICT_DISTINCT,
                    content_hash_a=fingerprints[record_a],
                    content_hash_b=fingerprints[record_b],
                )
            )

    clusters: dict[int, list[Event]] = {}
    for i, event in enumerate(events):
        clusters.setdefault(find(i), []).append(event)

    merged: list[Event] = []
    superseded: list[Event] = []
    for cluster in clusters.values():
        winner = merge_cluster(cluster)
        merged.append(winner)
        superseded.extend(
            _mark_superseded(cluster, winner, decisions, pass_name, identify, now)
        )

    return DedupResult(events=merged, decisions=decisions, superseded=superseded)


def _mark_superseded(
    cluster: list[Event],
    winner: Event,
    decisions: list[MergeDecision],
    pass_name: str,
    identify: Callable[[Event], str | None],
    now: datetime | None,
) -> list[Event]:
    """Every event the merge absorbed, marked with what absorbed it.

    Keyed back to the decision that joined each loser to the cluster so the
    score on the row is the one actually measured, not the cluster's best or
    last. A loser with no recoverable decision still records the merge — losing
    the score is better than losing the fact.
    """
    if len(cluster) == 1:
        return []

    scores = {
        tuple(sorted((d.record_a, d.record_b))): d.score
        for d in decisions
        if d.verdict == VERDICT_MERGED
    }
    winner_id = identify(winner)

    losers = []
    for event in cluster:
        if event.event_id == winner.event_id:
            continue
        loser = copy.deepcopy(event)
        loser.superseded_by = winner.event_id
        loser.superseded_at = now
        loser.merged_by = pass_name
        loser_id = identify(event)
        if winner_id is not None and loser_id is not None:
            loser.merge_similarity = scores.get(tuple(sorted((winner_id, loser_id))))
        losers.append(loser)
    return losers


class DeduplicationEngine:
    """Deduplicate a list of Events using fuzzy matching (Pass 1).

    Pure — no I/O, no DB access. Uses a greedy union-find approach:
    events are assigned to clusters; each cluster merges into one canonical event.
    """

    def deduplicate(self, events: list[Event], config: DeduplicationConfig) -> DedupResult:
        """Merge duplicate events within the given list.

        Args:
            events: Normalized events from NormalizationEngine.
            config: Deduplication thresholds and windows.

        Returns:
            Deduplicated events (order not guaranteed), and every comparison
            this pass made — merged and rejected alike.
        """
        return cluster_and_merge(
            events,
            lambda a, b: _compare(a, b, config),
            _candidate_identity,
            _listing_text,
            pass_name="fuzzy",
            record_kind="candidate",
        )
