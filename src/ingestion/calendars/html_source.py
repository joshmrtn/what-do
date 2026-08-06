"""HTML event-listing source adapter.

Complements the ICS adapter rather than replacing it. A calendar feed tends to
run weeks ahead for the handful of venues that maintain one; a listing page
covers far more venues but only the days it currently displays. Running both and
letting deduplication reconcile the overlap gives breadth and lookahead at once.

The listing also carries per-event links to venue pages, which calendar feeds
routinely omit.
"""

from __future__ import annotations

import hashlib
import zoneinfo
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import requests

from src.config import FeedConfig
from src.ingestion.calendars.fetching import fetch_document
from src.ingestion.calendars.listing import ListingEntry, parse_listing
from src.ingestion.source import IngestionSource
from src.models.event_candidate import EventCandidate


class HtmlListingSource(IngestionSource):
    """Fetches event candidates from a date-grouped HTML listing page."""

    def __init__(
        self,
        config: FeedConfig,
        db_path: Path | str,
        tzname: str,
        session: requests.Session | None = None,
        get_now: Callable[[], datetime] = datetime.now,
        logger: Any = None,
    ) -> None:
        """
        Args:
            config: The feed's name, URL, and politeness settings.
            db_path: Database holding the response cache.
            tzname: Zone the listing's wall-clock times are written in.
            session: Injected HTTP session.
            get_now: Injected clock.
            logger: Structured logger. Optional.
        """
        self._config = config
        self._db_path = db_path
        self._tz = zoneinfo.ZoneInfo(tzname)
        self._session = session or requests.Session()
        self._get_now = get_now
        self._logger = logger

    def fetch(self) -> list[EventCandidate]:
        """Fetch and parse the listing, skipping the network when polite to.

        Returns:
            One EventCandidate per placeable listing line, in page order.
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

        # The page's headings are local dates, so "today" must be local too —
        # read in UTC, an evening listing would resolve to the following day.
        today = self._get_now().astimezone(self._tz).date()
        entries = parse_listing(body, today=today, logger=self._logger)

        # Listing pages repeat an event now and then. Since identity is derived
        # from content, those collapse to one id — emit one candidate to match,
        # rather than two objects that claim to be the same thing.
        seen: dict[str, EventCandidate] = {}
        for entry in entries:
            candidate = self._to_candidate(entry)
            if candidate.id in seen:
                self._log(
                    f"Collapsing a repeated listing line from {self._config.name}: "
                    f"{entry.title!r} at {entry.venue!r}"
                )
                continue
            seen[candidate.id] = candidate

        return list(seen.values())

    def _to_candidate(self, entry: ListingEntry) -> EventCandidate:
        """Map one listing line to a candidate."""
        return EventCandidate(
            id=self._derive_id(entry),
            source=self._config.name,
            source_type=self._config.source_type,
            title=entry.title,
            description=(f"Category: {entry.category}" if entry.category else None),
            venue=entry.venue,
            location=entry.city,
            url=entry.url,
            # The page states a wall-clock time with no zone. Localising here
            # keeps a naive value out of ingestion, which compares against an
            # aware clock.
            start_time=entry.start.replace(tzinfo=self._tz),
            # The listing declares no end time; inventing one would be a guess.
            end_time=None,
            # No announcement date exists, and this field drives the ingestion
            # lookback discard.
            raw_published_at=None,
            discovered_at=self._get_now(),
        )

    def _derive_id(self, entry: ListingEntry) -> str:
        """Build a stable id, since the listing offers no identifier of its own.

        Derived from content rather than generated, so a nightly refetch updates
        the same rows instead of duplicating every event it has ever seen.
        """
        material = "|".join(
            (
                self._config.name,
                entry.start.isoformat(),
                entry.title,
                entry.venue or "",
                entry.city or "",
            )
        )
        digest = hashlib.sha256(material.encode("utf-8")).hexdigest()[:16]
        return f"{self._config.name}:{digest}"

    def _log(self, message: str) -> None:
        if self._logger is not None:
            self._logger.info(message, component="html_source", duration_ms=0)
