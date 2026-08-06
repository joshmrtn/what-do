"""Persistence for conditional HTTP requests.

Exists so politeness survives a process restart. The batch runs unattended
overnight, but a person debugging it may run it several times in an evening —
without persisted validators, every one of those runs would be a full download
of somebody else's server.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass(frozen=True)
class CachedResponse:
    """A previously fetched body with whatever validators the server offered."""

    body: str
    etag: str | None
    last_modified: str | None
    fetched_at: datetime


def read_cache(db_path: Path | str, url: str) -> CachedResponse | None:
    """Read the cached response for a URL, if one has been stored.

    Args:
        db_path: Path to the SQLite database.
        url: The URL used as the cache key.

    Returns:
        The cached response, or None when the URL has never been fetched.
    """
    conn = sqlite3.connect(db_path)
    try:
        row = conn.execute(
            "SELECT body, etag, last_modified, fetched_at FROM http_cache WHERE url = ?",
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


def write_cache(
    db_path: Path | str,
    url: str,
    *,
    body: str,
    etag: str | None,
    last_modified: str | None,
    fetched_at: datetime,
) -> None:
    """Store a fetched response, replacing any earlier entry for the same URL."""
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            """INSERT OR REPLACE INTO http_cache
               (url, etag, last_modified, body, fetched_at)
               VALUES (?, ?, ?, ?, ?)""",
            (url, etag, last_modified, body, fetched_at.isoformat()),
        )
        conn.commit()
    finally:
        conn.close()
