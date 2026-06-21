"""FailoverChain — tries ingestion sources in priority order, returns first success."""

from __future__ import annotations

from typing import Any

from src.ingestion.source import IngestionSource
from src.models.event_candidate import EventCandidate


class FailoverChain:
    """Tries sources in registration order; returns candidates from the first success.

    If a source raises any exception it is logged and the next source is tried.
    If all sources fail, returns an empty list.
    """

    def __init__(self, sources: list[IngestionSource], logger: Any) -> None:
        self._sources = sources
        self._logger = logger

    def fetch_all(self) -> list[EventCandidate]:
        """Return candidates from the first source that succeeds."""
        for source in self._sources:
            try:
                return source.fetch()
            except Exception as exc:
                self._logger.error(
                    f"Source {source.__class__.__name__} failed: {exc}",
                    component="failover",
                    duration_ms=0,
                )
        return []
