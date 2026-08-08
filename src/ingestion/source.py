"""IngestionSource ABC — contract for all event source adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod

from src.models.event_candidate import EventCandidate


class IngestionSource(ABC):
    """Base class for all event source adapters."""

    @property
    def source_name(self) -> str:
        """What to call this source in a report.

        Config-declared adapters override this with the feed's configured name,
        so a diagnostic run can say `do617_koto: 0` rather than naming the class
        that four other feeds also use.

        Deliberately not called `name`: `unittest.mock` special-cases that
        attribute, so a `MagicMock(spec=IngestionSource)` would hand back a mock
        instead of a string and quietly poison any report built from it.
        """
        return self.__class__.__name__

    @abstractmethod
    def fetch(self) -> list[EventCandidate]:
        """Fetch raw event candidates from this source.

        Returns:
            List of EventCandidate objects. Raises on unrecoverable error.
        """
