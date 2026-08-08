"""RSS feed source adapter.

Handles everything an RSS event source has in common — fetching politely,
parsing the feed, windowing by the night, and mapping to candidates — and leaves
exactly one thing to each site: reading an event out of an item.

That split exists because RSS has no event date. A feed item carries a headline,
a body and a *publication* date, so where the show's date lives is a convention
each site invents. Subclasses implement `interpret`, and anything they cannot
place is dropped rather than guessed: a wrong date is worse than a missing event.
"""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
from datetime import datetime, time, timedelta
from pathlib import Path
from typing import Any, Callable
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import requests

from src.config import DEFAULT_DAY_STARTS_AT, DEFAULT_HORIZON_DAYS, FeedConfig
from src.ingestion.calendars.fetching import fetch_document
from src.ingestion.rss import RssItem, parse_rss
from src.ingestion.source import IngestionSource
from src.models.event_candidate import EventCandidate
from src.models.timing import EXACT
from src.utils.nights import night_start


@dataclass(frozen=True)
class RssEvent:
    """What a site's own convention says an item means."""

    title: str
    #: Timezone-aware, because a feed states no zone of its own.
    start: datetime
    timing: str = EXACT
    venue: str | None = None
    description: str | None = None


class RssFeedSource(IngestionSource):
    """Fetches event candidates from an RSS feed. Subclass to interpret items."""

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
            config: The feed's name, URL, and politeness settings.
            db_path: Database holding the response cache.
            session: Injected HTTP session.
            get_now: Injected clock.
            logger: Structured logger. Optional.
            timezone_name: Zone the night window is reckoned in, and the zone a
                subclass should place a bare wall clock in.
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
    def zone(self) -> ZoneInfo:
        """The zone a subclass should place its wall-clock times in."""
        return self._zone

    @property
    def day_starts_at(self) -> time:
        """Where a subclass should place an event whose hour was never published."""
        return self._day_starts_at

    @abstractmethod
    def interpret(self, item: RssItem) -> RssEvent | None:
        """Read one feed item as an event.

        Args:
            item: The parsed `<item>`.

        Returns:
            The event the item describes, or None if it cannot be placed on a
            date — which is a normal outcome, not an error.
        """

    def fetch(self) -> list[EventCandidate]:
        """Fetch the feed and return the events inside the window.

        Returns:
            One EventCandidate per interpretable item inside the window, in
            feed order.

        Raises:
            ValueError: If the response is not a parseable feed.
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

        candidates: list[EventCandidate] = []
        unplaceable = 0

        for item in parse_rss(body):
            event = self.interpret(item)
            if event is None:
                unplaceable += 1
                continue
            if not floor <= event.start < ceiling:
                continue
            candidates.append(self._to_candidate(item, event))

        if unplaceable and self._logger is not None:
            self._logger.info(
                f"{self._config.name}: {unplaceable} item(s) carried no readable date",
                component="rss",
                duration_ms=0,
            )

        return candidates

    def _to_candidate(self, item: RssItem, event: RssEvent) -> EventCandidate:
        return EventCandidate(
            id=f"{self._config.name}:{item.guid}",
            source=self._config.name,
            source_type=self._config.source_type,
            title=event.title,
            description=event.description,
            venue=event.venue or self._config.venue,
            location=self._config.city,
            url=item.link,
            start_time=event.start,
            end_time=None,
            timing=event.timing,
            # The one field pubDate legitimately answers: when the show was
            # announced. Reading it as a start would file a June show in May.
            raw_published_at=item.published_at,
            discovered_at=self._get_now(),
        )


def _zone_of(name: str) -> ZoneInfo:
    try:
        return ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError):
        return ZoneInfo("UTC")
