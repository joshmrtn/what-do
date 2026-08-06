"""Polite conditional fetching, shared by every calendar-style source.

Public calendars and listing pages are somebody else's server. A nightly job
that redownloads unconditionally, or that hammers on a hand re-run, is a
nightly job that gets blocked — so this is the only way those sources fetch.

Three guarantees: at most one request per configured interval, conditional
requests that let the server answer 304, and no retry loop on failure.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable

import requests

from src.storage.http_cache import read_cache, write_cache

#: Identifies the project rather than impersonating a browser.
USER_AGENT = "what-do/1.0 (local event aggregator; nightly batch)"

#: Seconds before a request is abandoned. No retry follows.
REQUEST_TIMEOUT = 30


def fetch_document(
    url: str,
    *,
    session: requests.Session,
    db_path: Path | str,
    get_now: Callable[[], datetime],
    min_fetch_interval_hours: float,
    label: str,
    logger: Any = None,
) -> str:
    """Fetch a document, skipping the network whenever politeness allows.

    Args:
        url: Document to fetch.
        session: Injected HTTP session, so tests never reach the network.
        db_path: Database holding the response cache.
        get_now: Injected clock.
        min_fetch_interval_hours: Floor between real requests. Zero disables it.
        label: Source name, used in log messages.
        logger: Structured logger. Optional.

    Returns:
        The document body, from cache when refetching would be impolite.
    """
    now = get_now()
    cached = read_cache(db_path, url)

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
        write_cache(
            db_path,
            url,
            body=cached.body,
            etag=cached.etag,
            last_modified=cached.last_modified,
            fetched_at=now,
        )
        return cached.body

    body = response.text
    write_cache(
        db_path,
        url,
        body=body,
        etag=response.headers.get("ETag"),
        last_modified=response.headers.get("Last-Modified"),
        fetched_at=now,
    )
    return body


def _within_interval(fetched_at: datetime, now: datetime, hours: float) -> bool:
    """True while the politeness floor since the last fetch has not elapsed."""
    if hours <= 0:
        return False
    return now - fetched_at < timedelta(hours=hours)


def _log(logger: Any, message: str) -> None:
    if logger is not None:
        logger.info(message, component="fetching", duration_ms=0)
