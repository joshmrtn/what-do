"""Do617 venue-page source adapter.

Fetches one configured venue's page rather than crawling the site's date
listings: four or five requests a night, scoped to venues the user chose, versus
a crawl per day of the horizon for the same coverage.

Its reason for existing is the venues nothing else reaches. Gulu-Gulu Café
publishes a month of live music here while its own Squarespace feed returns zero
items. Koto and Bit Bar have valid venue pages that currently list nothing at
all — configured anyway, because they cost one request and start producing the
day Do617 has them.

Unlike every other listing adapter, no time is inferred: `startDate` carries the
offset, so a candidate's start is exactly what the source stated.
"""

from __future__ import annotations

from datetime import datetime, time, timedelta
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urljoin
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import requests

from src.config import DEFAULT_DAY_STARTS_AT, DEFAULT_HORIZON_DAYS, FeedConfig
from src.ingestion.aggregators.do617_listing import Do617Event, parse_do617
from src.ingestion.calendars.fetching import fetch_document
from src.ingestion.source import IngestionSource
from src.models.event_candidate import EventCandidate
from src.models.timing import EXACT
from src.utils.nights import night_start

#: Never walk further than this, whatever the listing claims. A pagination bug
#: on somebody else's server must not become an unbounded crawl on ours.
_DEFAULT_MAX_PAGES = 6

#: Categories whose slug does not read as English on its own.
_CATEGORY_LABELS = {"food-drink": "food & drink"}


class Do617VenueSource(IngestionSource):
    """Fetches event candidates from one Do617 venue page."""

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
        max_pages: int = _DEFAULT_MAX_PAGES,
    ) -> None:
        """
        Args:
            config: The venue page's name, URL, and politeness settings.
            db_path: Database holding the response cache.
            session: Injected HTTP session.
            get_now: Injected clock.
            logger: Structured logger. Optional.
            timezone_name: Zone the night window is reckoned in. The events
                themselves state their own offset.
            horizon_days: How far ahead to keep events.
            day_starts_at: Local time one night gives way to the next.
            max_pages: Hard cap on the pagination walk.
        """
        self._config = config
        self._db_path = db_path
        self._session = session or requests.Session()
        self._get_now = get_now
        self._logger = logger
        self._zone = _zone_of(timezone_name)
        self._horizon_days = horizon_days
        self._day_starts_at = day_starts_at
        self._max_pages = max_pages

    @property
    def source_name(self) -> str:
        """The feed's configured name, so a report can name the feed not the class."""
        return self._config.name

    def fetch(self) -> list[EventCandidate]:
        """Walk the venue's pages for as long as the horizon reaches.

        Returns:
            One EventCandidate per listed event inside the window, in page order.
        """
        floor = night_start(self._get_now(), self._day_starts_at, self._zone)
        ceiling = floor + timedelta(days=self._horizon_days)

        candidates: list[EventCandidate] = []
        seen: set[str] = set()
        url: str | None = self._config.url

        for page_number in range(1, self._max_pages + 1):
            if url is None:
                break

            page = parse_do617(self._read_page(url, page_number), logger=self._logger)

            fresh = [event for event in page.events if event.permalink not in seen]
            if page.events and not fresh:
                # Sites commonly serve page 1 for any out-of-range page number.
                # Stable ids mean the duplicates would be harmless downstream,
                # but the requests are not, and the walk would run to the cap.
                self._log(f"{self._config.name} page {page_number} repeated earlier events")
                break

            seen.update(event.permalink for event in fresh)
            candidates.extend(
                self._to_candidate(event) for event in fresh if floor <= event.start < ceiling
            )

            # Ascending order means a page whose *last* event is past the
            # horizon has already crossed it, so every later page is beyond it
            # too. Testing the first event instead would keep walking whenever a
            # page merely straddles the boundary — which the first one usually does.
            if fresh and fresh[-1].start >= ceiling:
                break

            url = urljoin(self._config.url, page.next_page_url) if page.next_page_url else None

        return candidates

    def _log(self, message: str) -> None:
        if self._logger is not None:
            self._logger.info(message, component="do617", duration_ms=0)

    def _read_page(self, url: str, page_number: int) -> str:
        return fetch_document(
            url,
            session=self._session,
            db_path=self._db_path,
            get_now=self._get_now,
            min_fetch_interval_hours=self._config.min_fetch_interval_hours,
            label=f"{self._config.name} page {page_number}",
            logger=self._logger,
        )

    def _to_candidate(self, event: Do617Event) -> EventCandidate:
        """Map one listed event, keeping the offset the markup stated."""
        return EventCandidate(
            id=f"{self._config.name}:{event.permalink}",
            source=self._config.name,
            source_type=self._config.source_type,
            title=event.title,
            description=_describe(event),
            # An aggregator lists many venues, so the card is the authority and
            # the configured venue only fills in for a card that names none.
            venue=event.venue or self._config.venue,
            location=event.city or self._config.city,
            url=urljoin(self._config.url, event.permalink),
            start_time=event.start,
            end_time=event.end,
            # The offset is in the markup, so there is nothing to place.
            timing=EXACT,
            # Deliberately unset, as for the other listings: a listing carries
            # no announcement date, and that field is what the lookback discards on.
            raw_published_at=None,
            discovered_at=self._get_now(),
        )


def _describe(event: Do617Event) -> str | None:
    """The site's own category, for extraction to read."""
    if not event.category:
        return None

    return _CATEGORY_LABELS.get(event.category, event.category.replace("-", " "))


def _zone_of(name: str) -> ZoneInfo:
    try:
        return ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError):
        return ZoneInfo("UTC")
