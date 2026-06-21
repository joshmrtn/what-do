"""Apify Instagram scraper adapter."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Callable

import requests

from src.ingestion.source import IngestionSource
from src.models.event_candidate import EventCandidate

_APIFY_BASE = "https://api.apify.com/v2"


class ApifyAdapter(IngestionSource):
    """Fetches Instagram posts via the Apify platform."""

    def __init__(
        self,
        api_key: str,
        handles: list[str],
        session: requests.Session | None = None,
        get_now: Callable[[], datetime] = datetime.now,
    ) -> None:
        self._api_key = api_key
        self._handles = handles
        self._session = session or requests.Session()
        self._get_now = get_now

    def fetch(self) -> list[EventCandidate]:
        """Fetch recent posts for configured handles via Apify."""
        url = f"{_APIFY_BASE}/acts/apify~instagram-scraper/runs"
        response = self._session.get(
            url,
            params={"token": self._api_key, "usernames": ",".join(self._handles)},
        )
        response.raise_for_status()
        posts: list[dict[str, Any]] = response.json()
        return [self._to_candidate(p) for p in posts]

    def _to_candidate(self, post: dict[str, Any]) -> EventCandidate:
        raw_ts = post.get("timestamp")
        pub_at = datetime.fromisoformat(raw_ts).replace(tzinfo=timezone.utc) if raw_ts else None
        return EventCandidate(
            id=str(uuid.uuid4()),
            source=post.get("ownerUsername", ""),
            source_type="apify",
            url=post.get("url"),
            image_url=post.get("displayUrl"),
            raw_published_at=pub_at,
            description=post.get("caption"),
            venue=post.get("locationName"),
            discovered_at=self._get_now(),
        )
