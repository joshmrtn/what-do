"""IngestionSource ABC — contract for all event source adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod

from src.models.event_candidate import EventCandidate


class IngestionSource(ABC):
    """Base class for all event source adapters."""

    @abstractmethod
    def fetch(self) -> list[EventCandidate]:
        """Fetch raw event candidates from this source.

        Returns:
            List of EventCandidate objects. Raises on unrecoverable error.
        """
