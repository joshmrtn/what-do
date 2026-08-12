"""Polite conditional fetching, shared by every calendar-style source.

Public calendars and listing pages are somebody else's server. A nightly job
that redownloads unconditionally, or that hammers on a hand re-run, is a
nightly job that gets blocked — so this is the only way those sources fetch.

Three guarantees: at most one request per configured interval, conditional
requests that let the server answer 304, and no retry loop on failure.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Callable

import requests

from src.storage.protocols import HttpCache

#: Identifies the project rather than impersonating a browser.
USER_AGENT = "what-do/1.0 (local event aggregator; nightly batch)"

#: Seconds before a request is abandoned. No retry follows.
REQUEST_TIMEOUT = 30


def fetch_document(
    url: str,
    *,
    session: requests.Session,
    get_now: Callable[[], datetime],
    http_cache: HttpCache,
    min_fetch_interval_hours: float,
    label: str,
    logger: Any = None,
) -> str:
    """Fetch a document, skipping the network whenever politeness allows.

    Args:
        url: Document to fetch.
        session: Injected HTTP session, so tests never reach the network.
        get_now: Injected clock.
        http_cache: Where conditional-request validators are stored. Required:
            a default that built its own SQLite cache meant a caller who forgot
            to inject silently got a database anyway, which is the failure the
            repository split exists to remove.
        min_fetch_interval_hours: Floor between real requests. Zero disables it.
        label: Source name, used in log messages.
        logger: Structured logger. Optional.

    Returns:
        The document body, from cache when refetching would be impolite.
    """
    cache = http_cache

    now = get_now()
    cached = cache.get(url)

    if cached is not None and _within_interval(
        cached.fetched_at, now, min_fetch_interval_hours
    ):
        _log(
            logger,
            f"Reusing the cached copy for {label}: last fetched "
            f"{cached.fetched_at.isoformat()}",
        )
        return cached.body

    headers = {"User-Agent": USER_AGENT}
    if cached is not None:
        if cached.etag:
            headers["If-None-Match"] = cached.etag
        if cached.last_modified:
            headers["If-Modified-Since"] = cached.last_modified

    response = session.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()

    if response.status_code == 304 and cached is not None:
        _log(logger, f"{label} is unchanged; serving the cached copy")
        cache.put(
            url,
            body=cached.body,
            etag=cached.etag,
            last_modified=cached.last_modified,
            fetched_at=now,
        )
        return cached.body

    body = response.text
    cache.put(
        url,
        body=body,
        etag=response.headers.get("ETag"),
        last_modified=response.headers.get("Last-Modified"),
        fetched_at=now,
    )
    return body


def _within_interval(fetched_at: datetime, now: datetime, hours: float) -> bool:
    """True while the politeness floor since the last fetch has not elapsed.

    Both sides are read as UTC when they state no zone. The cache is our own
    write and the batch clock is UTC, but a row left by an older naive clock
    would otherwise raise here — and since every configured source fetches
    through this function, one legacy row failed all seventeen at once.
    """
    if hours <= 0:
        return False
    return _as_utc(now) - _as_utc(fetched_at) < timedelta(hours=hours)


def _as_utc(value: datetime) -> datetime:
    """Read a bare timestamp as UTC, leaving one that states its zone alone."""
    return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)


def _log(logger: Any, message: str) -> None:
    if logger is not None:
        logger.info(message, component="fetching", duration_ms=0)
