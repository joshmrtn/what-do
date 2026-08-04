"""Event data model — normalized canonical event."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from typing import TYPE_CHECKING, Any

from src.models.tag import Tag

if TYPE_CHECKING:
    from src.scoring.similarity import SimilarityResult


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
    image_bytes: bytes | None = None
    title: str | None = None
    venue: str | None = None
    description: str | None = None
    location: str | None = None
    start_time: datetime | None = None
    end_time: datetime | None = None
    tags: list[Tag] = field(default_factory=list)
    summary: str | None = None
    #: "indoor", "outdoor", or "unknown". Decides whether weather can adjust the
    #: score at all; "unknown" earns no adjustment in either direction.
    setting: str = "unknown"
    tag_embeddings: list[bytes] = field(default_factory=list)
    summary_embedding: bytes | None = None
    weather: dict[str, Any] | None = None
    astronomical_data: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    similarity: "SimilarityResult | None" = None
