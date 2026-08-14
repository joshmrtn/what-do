"""Deduplication engine — Pass 2 (semantic, post-embedding).

Catches duplicates that fuzzy title matching missed: the same event described
in different words by different sources. Runs on the output of Pass 1, so a
cluster already merged there arrives as a single event.
"""

from __future__ import annotations

from src.config import DeduplicationConfig
from src.models.event import Event
from src.normalization.deduplicator import (
    Comparison,
    DedupResult,
    cluster_and_merge,
    times_match,
    venues_match,
)
from src.utils.vectors import cosine, decode_vector


def _summary_similarity(a: Event, b: Event) -> float | None:
    """How alike two summaries are, or None when there is nothing to compare.

    A missing vector means unknown, not equal — two events with no summary
    embedding are never merged on that basis, and no decision is recorded
    either, because recording one would teach a model that an absent vector
    means "different".
    """
    if a.summary_embedding is None or b.summary_embedding is None:
        return None
    return cosine(
        decode_vector(a.summary_embedding), decode_vector(b.summary_embedding)
    )


def _compare(a: Event, b: Event, cfg: DeduplicationConfig) -> Comparison | None:
    """Semantic match on the summary, gated by the same structural guards as Pass 1.

    Summary similarity alone is not sufficient. A venue running karaoke every
    Thursday produces near-identical summaries week after week; without the
    venue and start-time guards, a whole season of recurrences would collapse
    into one event. Pass 2 therefore replaces only Pass 1's *title* comparison,
    keeping its structural constraints intact.
    """
    if not venues_match(a.venue, b.venue):
        return None
    if not times_match(a.start_time, b.start_time, cfg.time_window_hours):
        return None

    similarity = _summary_similarity(a, b)
    if similarity is None:
        return None
    return Comparison(score=similarity, duplicate=similarity >= cfg.semantic_threshold)


class SemanticDeduplicationEngine:
    """Merge events whose summaries mean the same thing (Pass 2).

    Pure — no I/O, no DB access. Requires embeddings, so it runs after the
    embedding stage and before similarity scoring.
    """

    def deduplicate(
        self, events: list[Event], config: DeduplicationConfig
    ) -> DedupResult:
        """Merge semantically duplicate events within the given list.

        Args:
            events: Embedded events, already through dedup Pass 1.
            config: Deduplication thresholds and windows.

        Returns:
            Deduplicated events (order not guaranteed), and every comparison
            this pass made. These records are stored, so they key on event ids,
            which are stable once an event exists.
        """
        return cluster_and_merge(
            events,
            lambda a, b: _compare(a, b, config),
            lambda event: event.event_id,
            pass_name="semantic",
            record_kind="event",
        )
