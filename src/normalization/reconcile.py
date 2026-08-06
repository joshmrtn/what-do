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
    index: dict[str, Event] = {}
    for event in stored:
        for candidate_id in event.source_event_candidates:
            index[candidate_id] = event

    reconciled: list[Event] = []
    stale: list[str] = []
    claimed: set[str] = set()

    for event in fresh:
        matches = _matches(event, index)
        # A stored id already adopted this run means a cluster split into two
        # fresh events. Letting both adopt it would collide on the primary key
        # and one would silently overwrite the other, so the later one stays new.
        matches = [m for m in matches if m.event_id not in claimed]

        if not matches:
            reconciled.append(event)
            continue

        winner, losers = matches[0], matches[1:]
        claimed.add(winner.event_id)
        stale.extend(loser.event_id for loser in losers)
        reconciled.append(_adopt(event, winner))

    return ReconcileResult(events=reconciled, stale_event_ids=stale)


def _matches(event: Event, index: dict[str, Event]) -> list[Event]:
    """Stored events sharing a candidate with `event`, richest first.

    A cluster that has grown to span events previously held apart yields more
    than one. The richest wins so the most expensive work survives; ties break
    on age and then id, so the choice cannot vary between runs.
    """
    found: dict[str, Event] = {}
    for candidate_id in event.source_event_candidates:
        match = index.get(candidate_id)
        if match is not None:
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
