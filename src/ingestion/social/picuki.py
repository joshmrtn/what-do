"""Picuki Instagram viewer adapter (failover for Apify)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Callable

from src.network.http import HttpFetcher

from src.ingestion.candidate_id import derive_candidate_id
from src.ingestion.source import IngestionSource
from src.models.event_candidate import EventCandidate
from src.models.source_type import PICUKI

_PICUKI_BASE = "https://www.picuki.com/api"


class PicukiAdapter(IngestionSource):
    """Fetches Instagram posts via the Picuki viewer."""

    def __init__(
        self,
        handles: list[str],
        fetcher: HttpFetcher,
        get_now: Callable[[], datetime] = datetime.now,
    ) -> None:
        """
        Args:
            handles: Instagram handles to read.
            fetcher: The polite conditional GET. **This adapter passed no
                timeout at all**, so an unresponsive mirror hung the whole batch
                indefinitely rather than failing its stage.
            get_now: Injected clock.
        """
        self._handles = handles
        self._fetcher = fetcher
        self._get_now = get_now

    def fetch(self) -> list[EventCandidate]:
        """Fetch recent posts for configured handles via Picuki."""
        candidates: list[EventCandidate] = []
        for handle in self._handles:
            username = handle.lstrip("@")
            posts: list[dict[str, Any]] = json.loads(
                self._fetcher.get(f"{_PICUKI_BASE}/profile/{username}", label="picuki")
            )
            candidates.extend(self._to_candidate(p, handle) for p in posts)
        return candidates

    def _to_candidate(self, post: dict[str, Any], source_handle: str) -> EventCandidate:
        raw_date = post.get("date")
        pub_at = datetime.fromisoformat(raw_date).replace(tzinfo=timezone.utc) if raw_date else None
        return EventCandidate(
            id=self._derive_id(post, source_handle),
            source=source_handle,
            source_type=PICUKI,
            url=post.get("link"),
            image_url=post.get("image"),
            raw_published_at=pub_at,
            description=post.get("text"),
            discovered_at=self._get_now(),
        )

    def _derive_id(self, post: dict[str, Any], source_handle: str) -> str:
        """Build a stable id so a nightly refetch updates the post's row."""
        natural_key = post.get("post_id") or post.get("link")
        if natural_key:
            return derive_candidate_id("picuki", natural_key)
        return derive_candidate_id(
            "picuki",
            source_handle,
            post.get("date"),
            post.get("text"),
        )
