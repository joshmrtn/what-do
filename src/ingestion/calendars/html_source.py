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
from datetime import datetime, timedelta
from typing import Any, Callable

from src.config import FeedConfig
from src.network.http import HttpFetcher
from src.ingestion.calendars.listing import ListingEntry, parse_listing
from src.ingestion.calendars.listing_category import category_metadata
from src.ingestion.source import IngestionSource
from src.models.event_candidate import EventCandidate
from src.models.tag import Tag


#: Titles whose activity is fully stated by the title itself. The listing's
#: `Karaoke & trivia` section also carries bingo nights and open mics, so the
#: match is on the title, never on the heading.
_AUTHORED_ACTIVITIES: dict[str, tuple[str, ...]] = {
    "karaoke": ("karaoke", "singing", "live music", "social"),
    "trivia": ("trivia", "quiz night", "game night", "social"),
}


class HtmlListingSource(IngestionSource):
    """Fetches event candidates from a date-grouped HTML listing page."""

    def __init__(
        self,
        config: FeedConfig,
        fetcher: HttpFetcher,
        tzname: str,
        get_now: Callable[[], datetime] = datetime.now,
        logger: Any = None,
    ) -> None:
        """
        Args:
            config: The feed's name, URL, and politeness settings.
            fetcher: The polite conditional GET every source fetches through.
            tzname: Zone the listing's wall-clock times are written in.
            get_now: Injected clock.
            logger: Structured logger. Optional.
        """
        self._config = config
        self._fetcher = fetcher
        self._tz = zoneinfo.ZoneInfo(tzname)
        self._get_now = get_now
        self._logger = logger

    @property
    def source_name(self) -> str:
        """The feed's configured name, so a report can name the feed not the class."""
        return self._config.name

    def fetch(self) -> list[EventCandidate]:
        """Fetch and parse the listing, skipping the network when polite to.

        Returns:
            One EventCandidate per placeable listing line, in page order.
        """
        body = self._fetcher.get(
            self._config.url,
            label=self._config.name,
            max_age=timedelta(hours=self._config.min_fetch_interval_hours),
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
            # The listing publishes no prose about an event — only the line
            # itself — so there is no description to record. The section
            # heading is a fact *about* the listing and travels as metadata.
            description=None,
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
            summary=_compose_summary(entry),
            tags=[Tag(text=t, weight=w) for t, w in _authored_tags(entry.title)],
            metadata=self._entry_metadata(entry),
        )

    def _entry_metadata(self, entry: ListingEntry) -> dict[str, Any]:
        """What the adapter states about this line, as data rather than prose."""
        metadata: dict[str, Any] = self._category_metadata(entry)
        metadata["authored_summary"] = True
        if _authored_tags(entry.title):
            metadata["authored_tags"] = True
        return metadata

    def _category_metadata(self, entry: ListingEntry) -> dict[str, str]:
        """The section heading, when it says something the title does not."""
        return category_metadata(entry.category)

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


def _compose_summary(entry: ListingEntry) -> str:
    """One sentence built from the fields the line already gave us.

    `7:00 PM - Trivia - The James - Essex` becomes "Trivia at The James in
    Essex". Composed from the parsed fields rather than by re-splitting the
    raw line, which we have already done once.
    """
    summary = entry.title
    if entry.venue:
        summary = f"{summary} at {entry.venue}"
    if entry.city:
        summary = f"{summary} in {entry.city}"
    return summary


def _authored_tags(title: str | None) -> tuple[tuple[str, float], ...]:
    """Tags for a title that names its whole activity, or nothing.

    Weights descend so the activity itself dominates, matching what extraction
    is asked to produce. A title the listing files under `Karaoke & trivia`
    that is neither — `DJ Bingo Night`, `Open Mic` — returns nothing and goes
    to the model like anything else.
    """
    if title is None:
        return ()
    lowered = title.casefold()
    for activity, tags in _AUTHORED_ACTIVITIES.items():
        if lowered == activity or lowered.startswith(f"{activity} "):
            weights = (1.0, 0.8, 0.6, 0.4)
            return tuple(zip(tags, weights))
    return ()
