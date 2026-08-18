"""In-memory weather cache — the single official fake."""

from __future__ import annotations

from datetime import date, datetime
from typing import Any


class InMemoryDayCache:
    """Holds forecasts in a dict keyed by day and place."""

    def __init__(self) -> None:
        self._entries: dict[tuple[str, float, float], tuple[dict[str, Any], datetime]] = {}

    def get(
        self,
        *,
        day: date,
        latitude: float,
        longitude: float,
        fresh_since: datetime,
    ) -> dict[str, Any] | None:
        """The cached forecast for a day and place, if it is fresh enough."""
        entry = self._entries.get((day.isoformat(), latitude, longitude))
        if entry is None:
            return None

        data, stamped = entry
        if (stamped.tzinfo is None) != (fresh_since.tzinfo is None):
            stamped = stamped.replace(tzinfo=fresh_since.tzinfo)
        return data if stamped >= fresh_since else None

    def put(
        self,
        *,
        day: date,
        latitude: float,
        longitude: float,
        data: dict[str, Any],
        now: datetime,
    ) -> None:
        """Store a forecast, replacing any entry for the same day and place."""
        self._entries[(day.isoformat(), latitude, longitude)] = (dict(data), now)
