"""Event data model — normalized canonical event."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from typing import TYPE_CHECKING, Any

from src.models.source_type import SYNTHETIC
from src.models.tag import Tag
from src.models.timing import EXACT

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
    #: How much is known about *when*. "exact" is a stated clock time,
    #: "all_day" is a date the source declared as all-day, and "unknown" is a
    #: date whose hour was never published. The last two share a placed start
    #: so the night window can position them, which is exactly why the
    #: distinction has to be recorded rather than inferred from the clock.
    timing: str = "exact"
    tag_embeddings: list[bytes] = field(default_factory=list)
    summary_embedding: bytes | None = None
    weather: dict[str, Any] | None = None
    #: The cached forecast this event's weather was taken from. The event keeps
    #: only the hour it was scored against; the full day series stays in
    #: `weather_cache` rather than being copied onto every event that shares a
    #: date — measured, that duplication was 2.6 MB against the cache's 66 KB.
    weather_cache_id: str | None = None
    #: Set only when a venue name resolves confidently to a discovered venue.
    #: Source venue strings are messy, so this stays None far more often than not.
    venue_id: str | None = None
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

    @property
    def states_a_time(self) -> bool:
        """Whether `start_time` is a time the source actually gave."""
        return self.timing == EXACT
    similarity: "SimilarityResult | None" = None

    def replace_tags(self, tags: list[Tag]) -> None:
        """Adopt a new set of tags, discarding the vectors of the old ones.

        A vector describes the tag it was built from, so tags cannot be replaced
        without invalidating them. Assigning `tags` directly leaves vectors that
        describe tags this event no longer has, and because the pairing is
        positional a shorter list silently drops the tail — which is why writing
        such an event is refused outright.

        Dropping them costs about a second a tag to rebuild, against the minutes
        an extraction costs. Keeping them cost a whole night's batch.
        """
        self.tags = list(tags)
        self.tag_embeddings = []

    def attach_tag_embeddings(self, vectors: list[bytes]) -> None:
        """Attach vectors for the current tags.

        Args:
            vectors: One vector per tag, in tag order.

        Raises:
            ValueError: If the vectors cannot be paired one-to-one with the
                tags. The event keeps whatever it already had — refusing a bad
                list must not also destroy a good one.
        """
        if len(vectors) != len(self.tags):
            raise ValueError(
                f"event {self.event_id} has {len(self.tags)} tag(s) but "
                f"{len(vectors)} vector(s) were offered"
            )
        self.tag_embeddings = list(vectors)

    def replace_summary(self, summary: str | None) -> None:
        """Adopt a new summary, discarding the vector of the old one.

        The same rule as `replace_tags`, and the same reason. Nothing can catch
        a stale summary vector after the fact — one blob has nothing to pair
        against — so it is invisible rather than fatal, which makes owning the
        replacement the only place it can be got right.
        """
        self.summary = summary
        self.summary_embedding = None

    @property
    def is_synthetic(self) -> bool:
        """Whether this event was authored from config rather than extracted.

        Its tags are hand-written, so a low tag count is an authoring choice
        rather than a failed extraction, and running LLM Pass 1 over it would
        overwrite what a human wrote.
        """
        return self.source_type == SYNTHETIC
