"""Event data model — normalized canonical event."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class Event:
    """A normalized, canonical event ready for enrichment and scoring."""

    event_id: str
    source_event_candidates: list[str]
    source_type: str
    created_at: datetime
    updated_at: datetime
    url: str | None = None
    image_url: str | None = None
    title: str | None = None
    venue: str | None = None
    description: str | None = None
    location: str | None = None
    start_time: datetime | None = None
    end_time: datetime | None = None
    tags: list[str] = field(default_factory=list)
    summary: str | None = None
    tag_embeddings: list[bytes] = field(default_factory=list)
    summary_embedding: bytes | None = None
    weather: dict | None = None
    astronomical_data: dict | None = None
    metadata: dict = field(default_factory=dict)
