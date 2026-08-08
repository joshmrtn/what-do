"""Source adapter for pages publishing schema.org events as JSON-LD.

One request, no pagination, and no inference of any kind: the markup states the
title, the start with its UTC offset, the venue, the address, and whether the
event has been cancelled. Where a site offers this, it is the best route it has,
better than its own RSS and better than an aggregator's listing of it.

Written for PEM, whose `/events` page publishes 97 events this way while the
museum appears in no feed at all and its Do617 page holds only an archive. It is
site-agnostic: any page with the same markup is a config entry.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, time, timedelta
from pathlib import Path
from typing import Any, Callable
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import requests

from src.config import DEFAULT_DAY_STARTS_AT, DEFAULT_HORIZON_DAYS, FeedConfig
from src.ingestion.aggregators.jsonld_listing import JsonLdEvent, parse_jsonld_events
from src.ingestion.calendars.fetching import fetch_document
from src.ingestion.source import IngestionSource
from src.models.event_candidate import EventCandidate
from src.models.timing import EXACT
from src.utils.nights import night_start


class JsonLdEventSource(IngestionSource):
    """Fetches event candidates from a page's schema.org JSON-LD."""

    def __init__(
        self,
        config: FeedConfig,
        db_path: Path | str,
        session: requests.Session | None = None,
        get_now: Callable[[], datetime] = datetime.now,
        logger: Any = None,
        timezone_name: str = "UTC",
        horizon_days: int = DEFAULT_HORIZON_DAYS,
        day_starts_at: time = DEFAULT_DAY_STARTS_AT,
    ) -> None:
        """
        Args:
            config: The page's name, URL, and politeness settings.
            db_path: Database holding the response cache.
            session: Injected HTTP session.
            get_now: Injected clock.
            logger: Structured logger. Optional.
            timezone_name: Zone the night window is reckoned in. The events
                themselves state their own offset.
            horizon_days: How far ahead to keep events.
            day_starts_at: Local time one night gives way to the next.
        """
        self._config = config
        self._db_path = db_path
        self._session = session or requests.Session()
        self._get_now = get_now
        self._logger = logger
        self._zone = _zone_of(timezone_name)
        self._horizon_days = horizon_days
        self._day_starts_at = day_starts_at

    @property
    def source_name(self) -> str:
        """The feed's configured name, so a report can name the feed not the class."""
        return self._config.name

    def fetch(self) -> list[EventCandidate]:
        """Fetch the page and return the events inside the window.

        Returns:
            One EventCandidate per listed event inside the window, in page order.
        """
        body = fetch_document(
            self._config.url,
            session=self._session,
            db_path=self._db_path,
            get_now=self._get_now,
            min_fetch_interval_hours=self._config.min_fetch_interval_hours,
            label=self._config.name,
            logger=self._logger,
        )

        floor = night_start(self._get_now(), self._day_starts_at, self._zone)
        ceiling = floor + timedelta(days=self._horizon_days)

        return [
            self._to_candidate(event)
            for event in parse_jsonld_events(body, logger=self._logger)
            if floor <= event.start < ceiling
        ]

    def _to_candidate(self, event: JsonLdEvent) -> EventCandidate:
        """Map one event, keeping the offset the markup stated."""
        return EventCandidate(
            id=f"{self._config.name}:{self._key_for(event)}",
            source=self._config.name,
            source_type=self._config.source_type,
            title=event.title,
            description=event.description,
            venue=event.venue or self._config.venue,
            location=event.address or self._config.city,
            url=event.url,
            start_time=event.start,
            end_time=event.end,
            # The offset is in the markup, so there is nothing to place.
            timing=EXACT,
            # Deliberately unset, as for the other listings: a listing carries
            # no announcement date, and that field is what the lookback discards on.
            raw_published_at=None,
            discovered_at=self._get_now(),
        )

    def _key_for(self, event: JsonLdEvent) -> str:
        """What identifies one *occurrence*, not one programme.

        The URL alone is not enough. A recurring programme keeps a single page
        across every date it runs — PEM's 97 listings resolve to 61 URLs — so
        keying on it collapses a season of a weekly drop-in into one candidate
        and loses every date but the last.

        A page omitting the URL still has to produce the same id every night, or
        each run adopts its events as new, hence a digest rather than anything
        generated.
        """
        moment = event.start.isoformat()
        if event.url:
            return f"{event.url}@{moment}"

        digest = hashlib.sha256(f"{event.title}|{moment}|{event.venue or ''}".encode())
        return digest.hexdigest()[:16]


def _zone_of(name: str) -> ZoneInfo:
    try:
        return ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError):
        return ZoneInfo("UTC")
