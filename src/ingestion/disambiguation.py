"""Handle disambiguation step (batch step 3a).

Classifies probationary candidate_entities as 'venue' or 'person' using an LLM provider,
then evaluates handle promotion.
"""

from __future__ import annotations

import sqlite3
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal


class DisambiguationProvider(ABC):
    """Classifies a social handle as 'venue' or 'person'."""

    @abstractmethod
    def classify(self, handle: str, context: str) -> Literal["venue", "person"]:
        """Classify a handle given surrounding context text.

        Args:
            handle: The social handle to classify (e.g. '@jazzclub').
            context: Surrounding post caption that mentioned the handle.

        Returns:
            'venue' or 'person'.
        """


class DisambiguationStep:
    """Batch step 3a: classify new probationary handles and update their state."""

    def __init__(
        self,
        db_path: Path,
        provider: DisambiguationProvider,
        logger: Any,
    ) -> None:
        self._db_path = db_path
        self._provider = provider
        self._logger = logger

    def run(self) -> None:
        """Classify all unclassified probationary handles."""
        conn = sqlite3.connect(self._db_path)
        try:
            rows = conn.execute(
                """SELECT id, handle, discovery_context
                   FROM candidate_entities
                   WHERE state = 'probationary' AND llm_classification IS NULL""",
            ).fetchall()

            now = datetime.now(timezone.utc).isoformat()
            for entity_id, handle, context in rows:
                try:
                    classification = self._provider.classify(
                        handle=handle,
                        context=context or "",
                    )
                except Exception as exc:
                    self._logger.error(
                        f"Disambiguation failed for {handle}: {exc}",
                        component="disambiguation",
                        duration_ms=0,
                    )
                    continue

                new_state = "discarded" if classification == "person" else "probationary"
                conn.execute(
                    """UPDATE candidate_entities
                       SET llm_classification = ?, state = ?, updated_at = ?
                       WHERE id = ?""",
                    (classification, new_state, now, entity_id),
                )

            conn.commit()
        finally:
            conn.close()
