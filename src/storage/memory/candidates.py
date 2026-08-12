"""In-memory `CandidateRepository` — the single official fake."""

from __future__ import annotations

from datetime import datetime

from src.models.event_candidate import EventCandidate


class InMemoryCandidateRepository:
    """Holds candidates in a dict keyed by id."""

    def __init__(self) -> None:
        self._by_id: dict[str, EventCandidate] = {}

    def save(self, candidates: list[EventCandidate]) -> None:
        """Insert candidates, replacing any stored under the same id."""
        for candidate in candidates:
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
