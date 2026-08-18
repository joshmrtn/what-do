"""SQLite-backed cache of one day's readings for one place.

Two providers key their answers identically — a forecast and an air-quality
reading are both "what this service said about this day, here" — so they share
an implementation and differ only in the table they write. A second copy of
this class would drift from the first, silently, because nothing compares them.
"""

from __future__ import annotations

import json
import uuid
from datetime import date, datetime
from pathlib import Path
from typing import Any

from src.storage.sqlite.connection import connect


class SqliteDayCache:
    """Reads and writes one day-and-place cache table."""

    def __init__(self, db_path: Path | str, *, table: str) -> None:
        """
        Args:
            db_path: Database file.
            table: Which cache to read and write. Required rather than
                defaulted: a default would silently point a second provider at
                the first one's rows, and the `UNIQUE (date, latitude,
                longitude)` constraint means they would overwrite each other.
        """
        self._db_path = db_path
        self._table = _checked_table(table)

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
                f"SELECT data, fetched_at FROM {self._table} "
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
                f"INSERT OR REPLACE INTO {self._table} "
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


#: The tables this class may address. An allowlist rather than a check for
#: dangerous characters: the table name is interpolated into SQL, so the only
#: safe rule is that it was one of ours to begin with.
_TABLES = frozenset({"weather_cache", "air_quality_cache"})


def _checked_table(table: str) -> str:
    """The table name, or a refusal naming what is allowed."""
    if table not in _TABLES:
        raise ValueError(
            f"Unknown day cache table {table!r}; expected one of {sorted(_TABLES)}"
        )
    return table
