"""Match freshly normalized events to stored ones so identity survives a run.

`Normalizer` mints a new uuid per event, so without this every night's events
are new rows: the events table doubles, extraction never amortises at roughly
three minutes an event, and ranking shows everything twice. Matching on shared
candidate ids restores the identity that normalization discards.

Pure and separately testable, so the orchestrator stays a sequencer.
"""

from __future__ import annotations

import copy
from typing import NamedTuple

from src.models.event import Event
from src.normalization.deduplicator import merge_cluster

#: Carried from a stored event onto its fresh counterpart. `weather` is
#: deliberately absent: `weather.cache_ttl_hours` exists so a nightly batch
#: rescores against a forecast issued that night, and a carried-forward forecast
#: would score an event found a week out on the day it was found, forever.
#: `setting` travels with the extraction output that produced it — without it,
#: `ExtractionStage` skips on the carried tags and an outdoor event stays
#: "unknown", silently earning no weather adjustment ever again.
_CARRIED_FIELDS = (
    "tags",
    "summary",
    "setting",
    "tag_embeddings",
    "summary_embedding",
    "astronomical_data",
)


class ReconcileResult(NamedTuple):
    """Reconciled events, and stored ids left with no event to belong to."""

    events: list[Event]
    stale_event_ids: list[str]


def reconcile(fresh: list[Event], stored: list[Event]) -> ReconcileResult:
    """Adopt stored identity and enrichment onto freshly normalized events.

    Args:
        fresh: Events from this run's normalization and dedup.
        stored: Events already persisted, carrying tags, embeddings, and
            enrichment from previous runs.

    Returns:
        The fresh events in their original order, each either adopting a stored
        event's id and enrichment or keeping its own, plus the ids of stored
        events superseded by a merge and safe to delete.
    """
    # candidate -> every stored event claiming it. A plain dict would keep only
    # the last owner, and a hidden owner is never reported stale, so it lingers
    # as a duplicate for the life of the database.
    index: dict[str, list[Event]] = {}
    for event in stored:
        for candidate_id in event.source_event_candidates:
            index.setdefault(candidate_id, []).append(event)

    # Group the fresh events by the stored event each one claims, keeping the
    # position of the first member so the result stays in the caller's order.
    groups: dict[str, list[Event]] = {}
    winners: dict[str, Event] = {}
    order: list[str | None] = []
    unmatched: dict[int, Event] = {}
    stale: set[str] = set()

    for position, event in enumerate(fresh):
        matches = _matches(event, index)
        if not matches:
            unmatched[position] = event
            order.append(None)
            continue

        winner = matches[0]
        stale.update(loser.event_id for loser in matches[1:])
        if winner.event_id not in groups:
            groups[winner.event_id] = []
            winners[winner.event_id] = winner
            order.append(winner.event_id)
        else:
            order.append(None)
        groups[winner.event_id].append(event)

    reconciled: list[Event] = []
    for position, key in enumerate(order):
        if key is None:
            if position in unmatched:
                reconciled.append(unmatched[position])
            continue
        members = groups[key]
        # More than one fresh event claiming the same stored event means dedup
        # pass 1 split what pass 2 merged on an earlier run. Pass 2 is the more
        # capable judge and we already paid for its verdict, so the stored
        # event is treated as the record of it rather than re-litigated.
        merged = members[0] if len(members) == 1 else merge_cluster(members)
        reconciled.append(_adopt(merged, winners[key]))

    # A winner cannot also be stale: it was adopted.
    stale.difference_update(winners)
    return ReconcileResult(events=reconciled, stale_event_ids=sorted(stale))


def _matches(event: Event, index: dict[str, list[Event]]) -> list[Event]:
    """Stored events sharing a candidate with `event`, richest first.

    A cluster that has grown to span events previously held apart yields more
    than one. The richest wins so the most expensive work survives; ties break
    on age and then id, so the choice cannot vary between runs — two fresh
    events landing on the same stored winner depends on that determinism.
    """
    found: dict[str, Event] = {}
    for candidate_id in event.source_event_candidates:
        for match in index.get(candidate_id, ()):
            found[match.event_id] = match

    return sorted(
        found.values(),
        key=lambda e: (-_enrichment_depth(e), e.created_at, e.event_id),
    )


def _enrichment_depth(event: Event) -> int:
    """How much of the expensive work this event already carries."""
    return sum(1 for name in _CARRIED_FIELDS if _is_populated(name, getattr(event, name)))


def _is_populated(name: str, value: object) -> bool:
    """Whether a carried field holds real content.

    `setting` is an enum whose absent value is a string, so it is the one field
    where a truthy value can still mean nothing was determined.
    """
    if name == "setting":
        return value != "unknown"
    return bool(value)


def _adopt(event: Event, stored: Event) -> Event:
    """Copy stored identity and enrichment onto a fresh event.

    The fresh event stays authoritative for scraped content — a retitled or
    rescheduled event should reflect what the source says now — and
    `created_at` comes from storage because it records when the event was first
    seen, not when it was last read.
    """
    adopted = copy.deepcopy(event)
    adopted.event_id = stored.event_id
    adopted.created_at = stored.created_at

    for name in _CARRIED_FIELDS:
        value = getattr(stored, name)
        if _is_populated(name, value):
            setattr(adopted, name, copy.deepcopy(value))

    return adopted
