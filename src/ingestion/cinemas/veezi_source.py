"""Veezi public ticketing source adapter.

Reads a cinema's public `sessions` page and maps each showing to an
EventCandidate. Like the calendar sources this is forward-looking and already
structured, so title, time and booking link come out without an LLM.

No credentials. The page is keyed by a `siteToken` that appears in the cinema's
own booking links, and one adapter therefore covers every Veezi cinema — the
token is part of the configured URL, not code.

Politeness matches the calendar adapters: one conditional request per
`min_fetch_interval_hours`, cached across process restarts, and no retry loop.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Callable
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import requests

from src.storage.protocols import HttpCache
from src.config import FeedConfig
from src.ingestion.calendars.fetching import fetch_document
from src.ingestion.cinemas.veezi_listing import VeeziSession, parse_sessions
from src.ingestion.source import IngestionSource
from src.models.event_candidate import EventCandidate


class VeeziSessionsSource(IngestionSource):
    """Fetches event candidates from a Veezi public sessions page."""

    def __init__(
        self,
        config: FeedConfig,
        http_cache: HttpCache,
        session: requests.Session | None = None,
        get_now: Callable[[], datetime] = datetime.now,
        logger: Any = None,
        timezone_name: str = "UTC",
    ) -> None:
        self._config = config
        self._http_cache = http_cache
        self._session = session or requests.Session()
        self._get_now = get_now
        self._logger = logger
        self._zone = _zone_of(timezone_name)

    @property
    def source_name(self) -> str:
        """The feed's configured name, so a report can name the feed not the class."""
        return self._config.name

    def fetch(self) -> list[EventCandidate]:
        """Fetch and parse the sessions page, skipping the network when polite to.

        Returns:
            One EventCandidate per showing, in page order. Bounding by event time
            is left to ingestion, which applies the same window to every source.
        """
        body = fetch_document(
            self._config.url,
            session=self._session,
            http_cache=self._http_cache,
            get_now=self._get_now,
            min_fetch_interval_hours=self._config.min_fetch_interval_hours,
            label=self._config.name,
            logger=self._logger,
        )

        today = self._get_now().astimezone(self._zone).date()
        sessions = parse_sessions(body, today, logger=self._logger)

        return [self._to_candidate(showing) for showing in sessions]

    def _to_candidate(self, showing: VeeziSession) -> EventCandidate:
        """Map one showing, localising its wall clock to the cinema's zone."""
        return EventCandidate(
            id=f"{self._config.name}:{showing.session_id}",
            source=self._config.name,
            source_type=self._config.source_type,
            title=showing.title,
            description=None,
            # The page names the film and nothing else; only config knows which
            # cinema this is.
            venue=self._config.venue,
            location=self._config.city,
            url=showing.url,
            start_time=showing.start.replace(tzinfo=self._zone),
            # Runtime is not published here. TMDb enrichment can supply it later
            # rather than the adapter inventing a duration.
            end_time=None,
            # Deliberately unset, as for the calendar sources: a showtime
            # listing carries no announcement date, and this field is what the
            # ingestion lookback discards on.
            raw_published_at=None,
            discovered_at=self._get_now(),
        )


def _zone_of(name: str) -> ZoneInfo:
    """Resolve the cinema's zone, falling back to UTC rather than failing a run."""
    try:
        return ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError):
        return ZoneInfo("UTC")
