"""Similarity pipeline stage.

Scores every event against one preference set, attaching the result for the
ranking engine to consume.
"""

from __future__ import annotations

from src.config import ScoringConfig
from src.models.event import Event
from src.scoring.preferences import PreferenceSet
from src.scoring.similarity import SimilarityEngine


class SimilarityStage:
    """Attaches a SimilarityResult to each event.

    Preferences are supplied already loaded, so the embedding cache is consulted
    once per run rather than once per event, and the stage itself stays free of
    I/O — it is deterministic given its inputs.

    Args:
        preferences: Loaded likes and dislikes.
        config: Scoring thresholds, gate shape, and domain mapping.
    """

    def __init__(self, preferences: PreferenceSet, config: ScoringConfig) -> None:
        self._preferences = preferences
        self._config = config
        self._engine = SimilarityEngine()

    def process(self, events: list[Event]) -> list[Event]:
        """Score each event.

        Events whose embeddings are missing score zero rather than raising —
        one unscorable event costs one recommendation, not the batch.

        Args:
            events: Embedded events.

        Returns:
            The same list, with `similarity` attached in place.
        """
        for event in events:
            event.similarity = self._engine.score(
                event, self._preferences, self._config
            )
        return events
