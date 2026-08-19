"""Apify Instagram scraper adapter."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Callable

from src.network.http import HttpFetcher

from src.ingestion.candidate_id import derive_candidate_id
from src.ingestion.source import IngestionSource
from src.models.event_candidate import EventCandidate
from src.models.source_type import APIFY
from src.utils.secret import Secret

APIFY_HOST = "api.apify.com"
_APIFY_BASE = f"https://{APIFY_HOST}/v2"


class ApifyAdapter(IngestionSource):
    """Fetches Instagram posts via the Apify platform."""

    def __init__(
        self,
        api_key: Secret,
        handles: list[str],
        fetcher: HttpFetcher,
        get_now: Callable[[], datetime] = datetime.now,
    ) -> None:
        """
        Args:
            api_key: Apify credential.
            handles: Instagram handles to read.
            fetcher: The polite conditional GET. Apify is **metered and paid**,
                so it is the one place a wasted call costs money.
            get_now: Injected clock.
        """
        self._api_key = api_key
        self._handles = handles
        self._fetcher = fetcher
        self._get_now = get_now

    def fetch(self) -> list[EventCandidate]:
        """Fetch recent posts for configured handles via Apify."""
        url = f"{_APIFY_BASE}/acts/apify~instagram-scraper/runs"
        usernames = ",".join(self._handles)
        # The token is deliberately absent from the cache key. A key is written
        # to `http_cache`, so keying on the query string as sent would put a
        # paid credential at rest in the database — which the fetcher refuses
        # to do by accident, and this is the deliberate alternative.
        posts: list[dict[str, Any]] = json.loads(
            self._fetcher.get(
                url,
                label="apify",
                params={
                    "token": self._api_key.expose_secret(),
                    "usernames": usernames,
                },
                cache_key=f"{url}?usernames={usernames}",
            )
        )
        return [self._to_candidate(p) for p in posts]

    def _to_candidate(self, post: dict[str, Any]) -> EventCandidate:
        raw_ts = post.get("timestamp")
        pub_at = datetime.fromisoformat(raw_ts).replace(tzinfo=timezone.utc) if raw_ts else None
        return EventCandidate(
            id=self._derive_id(post),
            source=post.get("ownerUsername", ""),
            source_type=APIFY,
            url=post.get("url"),
            image_url=post.get("displayUrl"),
            raw_published_at=pub_at,
            description=post.get("caption"),
            venue=post.get("locationName"),
            discovered_at=self._get_now(),
        )

    def _derive_id(self, post: dict[str, Any]) -> str:
        """Build a stable id so a nightly refetch updates the post's row."""
        natural_key = post.get("id") or post.get("url")
        if natural_key:
            return derive_candidate_id("apify", natural_key)
        return derive_candidate_id(
            "apify",
            post.get("ownerUsername"),
            post.get("timestamp"),
            post.get("caption"),
        )
