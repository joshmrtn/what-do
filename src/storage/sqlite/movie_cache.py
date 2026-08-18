"""SQLite-backed cache of what TMDb said about a title.

**Raw, not derived.** This holds the provider's answer, keyed by the question
asked. What we go on to conclude a film *is* belongs in `movie_metadata`, which
is a separate table and a separate decision — the cache can expire without
taking a conclusion with it.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from src.models.movie_lookup import MovieLookup
from src.storage.sqlite.connection import connect


class SqliteMovieCache:
    """Reads and writes `tmdb_responses`."""

    def __init__(self, db_path: Path | str) -> None:
        self._db_path = db_path

    def get(
        self, *, title_key: str, year: int | None, fresh_since: datetime
    ) -> MovieLookup | None:
        """The stored lookup for a title and year, if it is fresh enough.

        Args:
            fresh_since: Entries stamped before this are not served. Passed in
                rather than checked afterwards, so a stale read has no API.
        """
        conn = connect(self._db_path)
        try:
            row = conn.execute(
                "SELECT payload, is_miss, fetched_at FROM tmdb_responses "
                "WHERE title_key=? AND year=?",
                (title_key, _year_key(year)),
            ).fetchone()
        finally:
            conn.close()

        if row is None or not _is_fresh(row[2], fresh_since):
            return None
        if row[1]:
            return MovieLookup(metadata=None)
        metadata: dict[str, Any] = json.loads(row[0])
        return MovieLookup(metadata=metadata)

    def put(
        self,
        *,
        title_key: str,
        year: int | None,
        lookup: MovieLookup,
        now: datetime,
    ) -> None:
        """Store a lookup, replacing any earlier answer to the same question.

        A miss is stored as a row with `is_miss`, not as an absent row: an
        absent row is indistinguishable from never having asked, which is what
        makes an unrecognised title a request that repeats for ever.
        """
        conn = connect(self._db_path)
        try:
            conn.execute(
                "INSERT OR REPLACE INTO tmdb_responses "
                "(id, title_key, year, payload, is_miss, fetched_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    str(uuid.uuid4()),
                    title_key,
                    _year_key(year),
                    json.dumps(lookup.metadata) if lookup.found else "",
                    0 if lookup.found else 1,
                    now.isoformat(),
                ),
            )
            conn.commit()
        finally:
            conn.close()


def _year_key(year: int | None) -> str:
    """A stable key part for an absent year, distinct from any real one."""
    return "" if year is None else str(year)


def _is_fresh(stamp: str, fresh_since: datetime) -> bool:
    """Whether a stored stamp is at or after the freshness bound."""
    stored = datetime.fromisoformat(stamp)
    if (stored.tzinfo is None) != (fresh_since.tzinfo is None):
        stored = stored.replace(tzinfo=fresh_since.tzinfo)
    return stored >= fresh_since
