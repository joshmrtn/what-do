"""In-memory movie lookup cache — the single official fake."""

from __future__ import annotations

from datetime import datetime

from src.models.movie_lookup import MovieLookup


class InMemoryMovieCache:
    """Holds lookups in a dict keyed by canonical title and year."""

    def __init__(self) -> None:
        self._entries: dict[tuple[str, str], tuple[MovieLookup, datetime]] = {}

    def get(
        self, *, title_key: str, year: int | None, fresh_since: datetime
    ) -> MovieLookup | None:
        """The stored lookup for a title and year, if it is fresh enough."""
        entry = self._entries.get((title_key, _year_key(year)))
        if entry is None:
            return None
        lookup, stamped = entry
        return lookup if stamped >= fresh_since else None

    def put(
        self,
        *,
        title_key: str,
        year: int | None,
        lookup: MovieLookup,
        now: datetime,
    ) -> None:
        """Store a lookup, replacing any earlier entry for the same question."""
        self._entries[(title_key, _year_key(year))] = (lookup, now)


def _year_key(year: int | None) -> str:
    """A stable key part for an absent year.

    Empty string rather than a number: no year is a different question from any
    particular year, and it must not collide with one.
    """
    return "" if year is None else str(year)
