"""One day of readings for one place, as a cache strategy.

Shared by the forecast and the air-quality endpoint. Both answer the same shape
of question — *what did this service say about this day, here* — so they share
a strategy and differ only in the table behind them. Two copies of this would
drift, silently, because nothing compares them.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Any, Callable

from src.storage.protocols import DayCache


class DayReadingsCache:
    """A `CacheStrategy` bound to one day and one place.

    The key is the caller's, because the policy cannot know what identifies a
    request — it never sees a URL, let alone a latitude. The lifetime is
    config's, read from the host's policy. `None` is a declared `never`.
    """

    def __init__(
        self,
        cache: DayCache,
        *,
        day: date,
        latitude: float,
        longitude: float,
        ttl: timedelta | None,
        get_now: Callable[[], datetime],
    ) -> None:
        self._cache = cache
        self._day = day
        self._latitude = latitude
        self._longitude = longitude
        self._ttl = ttl
        self._get_now = get_now

    def get(self) -> dict[str, Any] | None:
        """The stored forecast while it is still fresh, otherwise None."""
        if self._ttl is None:
            return None
        return self._cache.get(
            day=self._day,
            latitude=self._latitude,
            longitude=self._longitude,
            fresh_since=self._get_now() - self._ttl,
        )

    def put(self, value: dict[str, Any]) -> None:
        """Store the forecast, stamped from the injected clock."""
        if self._ttl is None:
            return
        self._cache.put(
            day=self._day,
            latitude=self._latitude,
            longitude=self._longitude,
            data=value,
            now=self._get_now(),
        )
