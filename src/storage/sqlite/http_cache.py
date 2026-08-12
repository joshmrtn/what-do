"""SQLite-backed HTTP response cache."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from src.storage.sqlite.connection import connect
from src.storage.http_cache import CachedResponse


class SqliteHttpCache:
    """Reads and writes `http_cache`."""

    def __init__(self, db_path: Path | str) -> None:
        self._db_path = db_path

    def get(self, url: str) -> CachedResponse | None:
        """The cached response for a URL, or None if never fetched."""
        conn = connect(self._db_path)
        try:
            row = conn.execute(
                "SELECT body, etag, last_modified, fetched_at FROM http_cache "
                "WHERE url = ?",
                (url,),
            ).fetchone()
        finally:
            conn.close()

        if row is None:
            return None

        return CachedResponse(
            body=row[0],
            etag=row[1],
            last_modified=row[2],
            fetched_at=datetime.fromisoformat(row[3]),
        )

    def put(
        self,
        url: str,
        *,
        body: str,
        etag: str | None,
        last_modified: str | None,
        fetched_at: datetime,
    ) -> None:
        """Store a response, replacing any earlier entry for the same URL."""
        conn = connect(self._db_path)
        try:
            conn.execute(
                "INSERT OR REPLACE INTO http_cache "
                "(url, etag, last_modified, body, fetched_at) VALUES (?, ?, ?, ?, ?)",
                (url, etag, last_modified, body, fetched_at.isoformat()),
            )
            conn.commit()
        finally:
            conn.close()
