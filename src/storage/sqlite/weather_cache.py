"""SQLite-backed weather cache."""

from __future__ import annotations

import json
import uuid
from datetime import date, datetime
from pathlib import Path
from typing import Any

from src.storage.db import connect


class SqliteWeatherCache:
    """Reads and writes `weather_cache`."""

    def __init__(self, db_path: Path | str) -> None:
        self._db_path = db_path

    def get(
        self,
        *,
        day: date,
        latitude: float,
        longitude: float,
        fresh_since: datetime,
    ) -> dict[str, Any] | None:
        """The cached forecast for a day and place, if it is fresh enough.

        Args:
            fresh_since: Entries stamped before this are not served. Passed in
                rather than checked by the caller afterwards, so a forecast
                cannot go stale by someone forgetting to look.
        """
        conn = connect(self._db_path)
        try:
            row = conn.execute(
                "SELECT data, fetched_at FROM weather_cache "
                "WHERE date=? AND latitude=? AND longitude=?",
                (day.isoformat(), latitude, longitude),
            ).fetchone()
        finally:
            conn.close()

        if row is None or not _is_fresh(row[1], fresh_since):
            return None
        cached: dict[str, Any] = json.loads(row[0])
        return cached

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
        conn = connect(self._db_path)
        try:
            conn.execute(
                "INSERT OR REPLACE INTO weather_cache "
                "(id, date, latitude, longitude, data, fetched_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    str(uuid.uuid4()),
                    day.isoformat(),
                    latitude,
                    longitude,
                    json.dumps(data),
                    now.isoformat(),
                ),
            )
            conn.commit()
        finally:
            conn.close()


def _is_fresh(fetched_at: str, fresh_since: datetime) -> bool:
    """Whether a stored stamp is recent enough to serve.

    An unparseable stamp is treated as expired: refetching costs one request,
    while trusting it could serve a forecast of any age.
    """
    try:
        stamped = datetime.fromisoformat(fetched_at)
    except ValueError:
        return False

    # The stamp is written by the same clock that produces `fresh_since`, so a
    # mismatch means the clock changed shape. Compare on common ground rather
    # than raising.
    if (stamped.tzinfo is None) != (fresh_since.tzinfo is None):
        stamped = stamped.replace(tzinfo=fresh_since.tzinfo)

    return stamped >= fresh_since
