"""In-memory `CandidateRepository` — the single official fake."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime

from src.models.event_candidate import EventCandidate


class InMemoryCandidateRepository:
    """Holds candidates in a dict keyed by id."""

    def __init__(self) -> None:
        self._by_id: dict[str, EventCandidate] = {}

    def save(self, candidates: list[EventCandidate]) -> None:
        """Insert candidates, replacing any stored under the same id.

        The published fields are replaced wholesale, but `discovered_at` is
        kept from the row already held: a re-fetch is a fresh sighting of a
        listing we already met, not a new discovery of it (#27).
        """
        for candidate in candidates:
            existing = self._by_id.get(candidate.id)
            if existing is not None:
                candidate = replace(candidate, discovered_at=existing.discovered_at)
            self._by_id[candidate.id] = candidate

    def for_window(
        self, *, discovered_since: datetime, starting_after: datetime
    ) -> list[EventCandidate]:
        """Candidates still in scope for a run, ordered by discovery then id."""
        matching = [
            candidate
            for candidate in self._by_id.values()
            if candidate.discovered_at >= discovered_since
            or (candidate.start_time is not None and candidate.start_time >= starting_after)
        ]
        return sorted(matching, key=lambda c: (c.discovered_at, c.id))
