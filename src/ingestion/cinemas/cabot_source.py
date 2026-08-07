"""The Cabot listing source adapter.

Reads `thecabot.org/whats-on`, which paginates ten events to a page. The listing
is strictly ascending, so paging stops as soon as a page opens past the horizon
rather than walking all nine — typically half the requests, and never more than
`max_pages`.

Politeness matches the other listing adapters: each page is a separate
conditional request cached by its own URL, one per `min_fetch_interval_hours`,
and no retry loop.
"""

from __future__ import annotations

from datetime import datetime, time, timedelta
from pathlib import Path
from typing import Any, Callable
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import requests

from src.config import DEFAULT_DAY_STARTS_AT, DEFAULT_HORIZON_DAYS, FeedConfig
from src.ingestion.calendars.fetching import fetch_document
from src.ingestion.cinemas.cabot_listing import CabotEvent, parse_cabot
from src.ingestion.source import IngestionSource
from src.models.event_candidate import EventCandidate
from src.models.timing import EXACT, UNKNOWN
from src.utils.nights import night_start

#: The listing's fixed page size. `?posts_per_page` and `?per_page` are both
#: ignored by the site, so this cannot be raised.
_PAGE_SIZE = 10

#: Never walk further than this, whatever the listing claims. A pagination bug
#: on somebody else's server must not become an unbounded crawl on ours.
_DEFAULT_MAX_PAGES = 12


class CabotListingSource(IngestionSource):
    """Fetches event candidates from The Cabot's paginated listing."""

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
        self._config = config
        self._db_path = db_path
        self._session = session or requests.Session()
        self._get_now = get_now
        self._logger = logger
        self._zone = _zone_of(timezone_name)
        self._horizon_days = horizon_days
        self._day_starts_at = day_starts_at
        self._max_pages = max_pages

    def fetch(self) -> list[EventCandidate]:
        """Fetch as many listing pages as the horizon actually reaches.

        Returns:
            One EventCandidate per listed event inside the window, in page order.
        """
        now = self._get_now()
        floor = night_start(now, self._day_starts_at, self._zone)
        ceiling = floor + timedelta(days=self._horizon_days)
        today = now.astimezone(self._zone).date()

        candidates: list[EventCandidate] = []
        seen: set[str] = set()
        pages = self._max_pages

        for page in range(1, self._max_pages + 1):
            events, total = parse_cabot(
                self._read_page(page), today, logger=self._logger, want_total=True
            )
            if page == 1 and total:
                pages = min(self._page_count(total), self._max_pages)

            if not events:
                break

            fresh = [event for event in events if event.event_id not in seen]
            if not fresh:
                # Sites commonly serve page 1 for any out-of-range page number.
                # Stable ids mean the duplicates would be harmless downstream,
                # but the requests are not, and the walk would run to the cap.
                self._log(f"{self._config.name} page {page} repeated earlier events")
                break

            seen.update(event.event_id for event in fresh)
            candidates.extend(
                self._to_candidate(event)
                for event in fresh
                if self._localise(event.start) < ceiling
            )

            # Ascending order means a page opening past the horizon guarantees
            # every later page does too, so there is nothing left worth asking for.
            if self._localise(events[0].start) >= ceiling or page >= pages:
                break

        return candidates

    def _log(self, message: str) -> None:
        if self._logger is not None:
            self._logger.info(message, component="cabot", duration_ms=0)

    def _page_count(self, total: int) -> int:
        return max(1, -(-total // _PAGE_SIZE))

    def _read_page(self, page: int) -> str:
        url = self._config.url if page == 1 else f"{self._config.url.rstrip('/')}/page/{page}/"

        return fetch_document(
            url,
            session=self._session,
            db_path=self._db_path,
            get_now=self._get_now,
            min_fetch_interval_hours=self._config.min_fetch_interval_hours,
            label=f"{self._config.name} page {page}",
            logger=self._logger,
        )

    def _to_candidate(self, event: CabotEvent) -> EventCandidate:
        """Map one listed event, localising its wall clock to the venue's zone."""
        return EventCandidate(
            id=f"{self._config.name}:{event.event_id}",
            source=self._config.name,
            source_type=self._config.source_type,
            title=event.title,
            # Genres are the listing's own labels and say what kind of thing this
            # is, which is exactly what extraction needs and cannot infer from a
            # title like "In Focus: Improv Lab".
            description=_describe(event),
            venue=_venue_of(event) or self._config.venue,
            location=self._config.city,
            url=event.url,
            start_time=self._localise(
                event.start
                if event.time_known
                else event.start.replace(
                    hour=self._day_starts_at.hour, minute=self._day_starts_at.minute
                )
            ),
            end_time=self._localise(event.end) if event.end else None,
            # The listing gave a date but no hour. Not the same as all day —
            # a drop-in has a start, nobody has published it.
            timing=EXACT if event.time_known else UNKNOWN,
            # Deliberately unset, as for the calendar sources: a listing carries
            # no announcement date, and that field is what the lookback discards on.
            raw_published_at=None,
            discovered_at=self._get_now(),
        )

    def _localise(self, when: datetime) -> datetime:
        return when.replace(tzinfo=self._zone)


def _describe(event: CabotEvent) -> str | None:
    """Genres plus any series or tour name, for extraction to read.

    Genres are the listing's own labels and say what kind of thing this is,
    which extraction cannot infer from a title like `In Focus: Improv Lab`.
    """
    parts = [", ".join(event.genres)] if event.genres else []
    if event.subtitle:
        # Off-site subtitles carry the venue *and* its address. The name is
        # already the venue, so what is left is the street — worth keeping,
        # since Off Cabot is a different building a few streets from the
        # theatre and "which building" decides whether you go.
        parts.append(_address_of(event) if event.off_site else event.subtitle)

    return " — ".join(p for p in parts if p) or None


def _address_of(event: CabotEvent) -> str:
    """The street part of an off-site subtitle, or the whole of it if unsplit."""
    _, separator, address = (event.subtitle or "").partition(" - ")

    return address.strip() if separator else (event.subtitle or "").strip()


def _venue_of(event: CabotEvent) -> str | None:
    """The venue for an off-site event, whose subtitle names it and its address.

    Only trusted when the listing flagged the event as off-site. The subtitle is
    overloaded — it holds `Off Cabot - 9 Wallis St, Beverly` for one event and
    `INDIGO PARK TOUR` for the next — so reading it unconditionally files a tour
    name as a venue.
    """
    if not event.off_site or not event.subtitle:
        return None

    return event.subtitle.split(" - ")[0].strip() or None


def _zone_of(name: str) -> ZoneInfo:
    """Resolve the venue's zone, falling back to UTC rather than failing a run."""
    try:
        return ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError):
        return ZoneInfo("UTC")
