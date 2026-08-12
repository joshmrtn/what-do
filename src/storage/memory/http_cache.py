"""In-memory HTTP response cache — the single official fake.

Lets an adapter's tests exercise the conditional-request path without a
database file, which is most of what those tests are actually about.
"""

from __future__ import annotations

from datetime import datetime

from src.storage.http_cache import CachedResponse


class InMemoryHttpCache:
    """Holds responses in a dict keyed by URL."""

    def __init__(self) -> None:
        self._by_url: dict[str, CachedResponse] = {}

    def get(self, url: str) -> CachedResponse | None:
        """The cached response for a URL, or None if never fetched."""
        return self._by_url.get(url)

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
        self._by_url[url] = CachedResponse(
            body=body, etag=etag, last_modified=last_modified, fetched_at=fetched_at
        )
