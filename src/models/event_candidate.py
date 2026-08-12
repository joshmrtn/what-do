"""EventCandidate data model — raw discovered event information."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from src.models.tag import Tag


@dataclass
class EventCandidate:
    """Raw event information as returned by an ingestion source adapter."""

    id: str
    source: str
    source_type: str
    discovered_at: datetime
    url: str | None = None
    image_url: str | None = None
    raw_published_at: datetime | None = None
    title: str | None = None
    description: str | None = None
    venue: str | None = None
    location: str | None = None
    start_time: datetime | None = None
    end_time: datetime | None = None
    #: How much is known about *when* — see `TIMINGS`. A source that gives a
    #: date but no hour says so here rather than letting a placed start read as
    #: a real one.
    timing: str = "exact"
    #: Structured facts a source states about an event that are not prose about
    #: it. A listing's section heading belongs here and not in `description`:
    #: putting a taxonomy label where event copy goes is what led LLM Pass 1 to
    #: read `Karaoke & trivia` as a claim about the event and tag every trivia
    #: night `karaoke`. Merged into `Event.metadata` by normalization.
    metadata: dict[str, Any] = field(default_factory=dict)
    #: A summary the source itself supports, composed rather than inferred.
    #: Terse sources state everything they know in one line; a model asked to
    #: summarise that can only repeat it or invent past it.
    summary: str | None = None
    #: Tags the adapter can author outright, for events whose title *is* the
    #: activity. Paired with `metadata["authored_tags"]`, which exempts the
    #: event from extraction entirely.
    tags: list[Tag] = field(default_factory=list)
