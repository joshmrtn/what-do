"""Event data model — normalized canonical event."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from typing import TYPE_CHECKING, Any

from src.models.source_type import SYNTHETIC
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
    #: Hash of the text LLM Pass 1 last ran over successfully. Set only on
    #: success, so it distinguishes three states the old `if tags` check
    #: conflated: set means done, absent means never ran, and a failure leaves
    #: it absent so the next run retries.
    extraction_input_hash: str | None = None
    #: Hash of the tags and summary the vectors were last built from. Because
    #: those are extraction's output, a re-extraction that changes them changes
    #: this too, so embeddings follow automatically rather than leaving vectors
    #: describing tags the event no longer has.
    embedding_input_hash: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    similarity: "SimilarityResult | None" = None

    @property
    def is_synthetic(self) -> bool:
        """Whether this event was authored from config rather than extracted.

        Its tags are hand-written, so a low tag count is an authoring choice
        rather than a failed extraction, and running LLM Pass 1 over it would
        overwrite what a human wrote.
        """
        return self.source_type == SYNTHETIC
