"""The value type for a cached HTTP response.

Reading and writing live in `storage/sqlite/http_cache.py`; the in-memory
implementation beside it is what lets an adapter exercise the
conditional-request path without a database file.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class CachedResponse:
    """A previously fetched body with whatever validators the server offered."""

    body: str
    etag: str | None
    last_modified: str | None
    fetched_at: datetime
