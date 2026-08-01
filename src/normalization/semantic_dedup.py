"""Deduplication engine — Pass 2 (semantic, post-embedding).

Catches duplicates that fuzzy title matching missed: the same event described
in different words by different sources. Runs on the output of Pass 1, so a
cluster already merged there arrives as a single event.
"""

from __future__ import annotations

from src.config import DeduplicationConfig
from src.models.event import Event
from src.normalization.deduplicator import cluster_and_merge, times_match, venues_match
from src.utils.vectors import cosine, decode_vector


def _summaries_match(a: Event, b: Event, threshold: float) -> bool:
    """True when both summary vectors exist and are similar enough.

    A missing vector means unknown, not equal — two events with no summary
    embedding are never merged on that basis.
    """
    if a.summary_embedding is None or b.summary_embedding is None:
        return False
    similarity = cosine(
        decode_vector(a.summary_embedding), decode_vector(b.summary_embedding)
    )
    return similarity >= threshold


def _are_duplicates(a: Event, b: Event, cfg: DeduplicationConfig) -> bool:
    """Semantic match on the summary, gated by the same structural guards as Pass 1.

    Summary similarity alone is not sufficient. A venue running karaoke every
    Thursday produces near-identical summaries week after week; without the
    venue and start-time guards, a whole season of recurrences would collapse
    into one event. Pass 2 therefore replaces only Pass 1's *title* comparison,
    keeping its structural constraints intact.
    """
    return (
        _summaries_match(a, b, cfg.semantic_threshold)
        and venues_match(a.venue, b.venue)
        and times_match(a.start_time, b.start_time, cfg.time_window_hours)
    )


class SemanticDeduplicationEngine:
    """Merge events whose summaries mean the same thing (Pass 2).

    Pure — no I/O, no DB access. Requires embeddings, so it runs after the
    embedding stage and before similarity scoring.
    """

    def deduplicate(
        self, events: list[Event], config: DeduplicationConfig
    ) -> list[Event]:
        """Merge semantically duplicate events within the given list.

        Args:
            events: Embedded events, already through dedup Pass 1.
            config: Deduplication thresholds and windows.

        Returns:
            Deduplicated list of Events (order not guaranteed).
        """
        return cluster_and_merge(
            events, lambda a, b: _are_duplicates(a, b, config)
        )
