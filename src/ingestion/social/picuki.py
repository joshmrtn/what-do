"""Picuki Instagram viewer adapter (failover for Apify)."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Callable

import requests

from src.ingestion.source import IngestionSource
from src.models.event_candidate import EventCandidate

_PICUKI_BASE = "https://www.picuki.com/api"


class PicukiAdapter(IngestionSource):
    """Fetches Instagram posts via the Picuki viewer."""

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
        """Fetch recent posts for configured handles via Picuki."""
        candidates: list[EventCandidate] = []
        for handle in self._handles:
            username = handle.lstrip("@")
            response = self._session.get(f"{_PICUKI_BASE}/profile/{username}")
            response.raise_for_status()
            posts: list[dict[str, Any]] = response.json()
            candidates.extend(self._to_candidate(p, handle) for p in posts)
        return candidates

    def _to_candidate(self, post: dict[str, Any], source_handle: str) -> EventCandidate:
        raw_date = post.get("date")
        pub_at = datetime.fromisoformat(raw_date).replace(tzinfo=timezone.utc) if raw_date else None
        return EventCandidate(
            id=str(uuid.uuid4()),
            source=source_handle,
            source_type="picuki",
            url=post.get("link"),
            image_url=post.get("image"),
            raw_published_at=pub_at,
            description=post.get("text"),
            discovered_at=self._get_now(),
        )
