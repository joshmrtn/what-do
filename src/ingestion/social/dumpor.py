"""Dumpor Instagram viewer adapter (second failover)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Callable

from src.network.http import HttpFetcher

from src.ingestion.candidate_id import derive_candidate_id
from src.ingestion.source import IngestionSource
from src.models.event_candidate import EventCandidate
from src.models.source_type import DUMPOR

_DUMPOR_BASE = "https://dumpor.com/api"


class DumporAdapter(IngestionSource):
    """Fetches Instagram posts via the Dumpor viewer."""

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
        """Fetch recent posts for configured handles via Dumpor."""
        candidates: list[EventCandidate] = []
        for handle in self._handles:
            username = handle.lstrip("@")
            posts: list[dict[str, Any]] = json.loads(
                self._fetcher.get(f"{_DUMPOR_BASE}/user/{username}", label="dumpor")
            )
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
            id=self._derive_id(post, source_handle),
            source=source_handle,
            source_type=DUMPOR,
            url=post.get("permalink"),
            image_url=post.get("display_url"),
            raw_published_at=pub_at,
            description=post.get("caption_text"),
            discovered_at=self._get_now(),
        )

    def _derive_id(self, post: dict[str, Any], source_handle: str) -> str:
        """Build a stable id so a nightly refetch updates the post's row."""
        natural_key = post.get("shortcode") or post.get("permalink")
        if natural_key:
            return derive_candidate_id("dumpor", natural_key)
        return derive_candidate_id(
            "dumpor",
            source_handle,
            post.get("taken_at_timestamp"),
            post.get("caption_text"),
        )
