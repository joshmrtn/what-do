"""Dumpor Instagram viewer adapter (second failover)."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Callable

import requests

from src.ingestion.source import IngestionSource
from src.models.event_candidate import EventCandidate

_DUMPOR_BASE = "https://dumpor.com/api"


class DumporAdapter(IngestionSource):
    """Fetches Instagram posts via the Dumpor viewer."""

    def __init__(
        self,
        handles: list[str],
        session: requests.Session | None = None,
        get_now: Callable[[], datetime] = datetime.now,
    ) -> None:
        self._handles = handles
        self._session = session or requests.Session()
        self._get_now = get_now

    def fetch(self) -> list[EventCandidate]:
        """Fetch recent posts for configured handles via Dumpor."""
        candidates: list[EventCandidate] = []
        for handle in self._handles:
            username = handle.lstrip("@")
            response = self._session.get(f"{_DUMPOR_BASE}/user/{username}")
            response.raise_for_status()
            posts: list[dict[str, Any]] = response.json()
            candidates.extend(self._to_candidate(p, handle) for p in posts)
        return candidates

    def _to_candidate(self, post: dict[str, Any], source_handle: str) -> EventCandidate:
        raw_ts = post.get("taken_at_timestamp")
        pub_at = (
            datetime.fromtimestamp(int(raw_ts), tz=timezone.utc)
            if raw_ts is not None
            else None
        )
        return EventCandidate(
            id=str(uuid.uuid4()),
            source=source_handle,
            source_type="dumpor",
            url=post.get("permalink"),
            image_url=post.get("display_url"),
            raw_published_at=pub_at,
            description=post.get("caption_text"),
            discovered_at=self._get_now(),
        )
