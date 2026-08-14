"""In-memory `CandidateRepository` — the single official fake."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone

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
        self, *, seen_since: datetime, starting_after: datetime
    ) -> list[EventCandidate]:
        """Candidates still in scope for a run, ordered by discovery then id.

        Split on what is known: a dated candidate while its event is still to
        come, an undated one while a source is still publishing it. Comparing
        aware datetimes, so the bounds' own zones do not matter here — they are
        canonicalised anyway, to keep this answering exactly what SQLite does.
        """
        after = starting_after.astimezone(timezone.utc)
        seen = seen_since.astimezone(timezone.utc)
        matching = [
            candidate
            for candidate in self._by_id.values()
            if (candidate.start_time is not None and candidate.start_time >= after)
            or (
                candidate.start_time is None
                and (candidate.last_seen_at or candidate.discovered_at) >= seen
            )
        ]
        return sorted(matching, key=lambda c: (c.discovered_at, c.id))
