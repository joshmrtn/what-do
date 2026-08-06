"""Public ICS calendar source adapter.

Reads a published .ics feed and maps its events to EventCandidates. Unlike the
social adapters this source is forward-looking and already structured, so venue,
city, start and end come out without an LLM ever being involved.

The adapter is deliberately polite: one conditional request per batch run, a
configurable floor between fetches that survives process restarts, and no retry
loop. Public calendars are somebody else's server, and a nightly job that
misbehaves is a nightly job that gets blocked.
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import requests

from src.config import FeedConfig
from src.ingestion.calendars.fetching import fetch_document
from src.ingestion.ics import VEvent, parse_ics
from src.ingestion.source import IngestionSource
from src.models.event_candidate import EventCandidate
from src.utils.html import html_to_text

#: `[Venue, City]` or `[Venue, City, Category]` leading a summary.
_PREFIX_RE = re.compile(r"^\s*\[(?P<prefix>[^\]]*)\]\s*(?P<title>.*)$", re.DOTALL)

#: Zero-width and other format characters that lead every summary in some feeds.
_INVISIBLE_RE = re.compile(r"[​‌‍﻿]")


class IcsCalendarSource(IngestionSource):
    """Fetches event candidates from a public ICS calendar feed."""

    def __init__(
        self,
        config: FeedConfig,
        db_path: Path | str,
        session: requests.Session | None = None,
        get_now: Callable[[], datetime] = datetime.now,
        logger: Any = None,
    ) -> None:
        self._config = config
        self._db_path = db_path
        self._session = session or requests.Session()
        self._get_now = get_now
        self._logger = logger

    def fetch(self) -> list[EventCandidate]:
        """Fetch and parse the calendar, skipping the network when it is polite to.

        Returns:
            One EventCandidate per usable VEVENT, in feed order.
        """
        body = self._read_feed()
        events = parse_ics(body, logger=self._logger)

        candidates = []
        for event in events:
            candidate = self._to_candidate(event)
            if candidate is not None:
                candidates.append(candidate)

        return candidates

    # ------------------------------------------------------------------
    # Fetching
    # ------------------------------------------------------------------

    def _read_feed(self) -> str:
        """Return the feed body, from cache when refetching would be impolite."""
        return fetch_document(
            self._config.url,
            session=self._session,
            db_path=self._db_path,
            get_now=self._get_now,
            min_fetch_interval_hours=self._config.min_fetch_interval_hours,
            label=self._config.name,
            logger=self._logger,
        )

    # ------------------------------------------------------------------
    # Mapping
    # ------------------------------------------------------------------

    def _to_candidate(self, event: VEvent) -> EventCandidate | None:
        """Map one VEVENT, or None when it cannot be used."""
        if (event.status or "").upper() == "CANCELLED":
            return None

        if not event.uid:
            self._log(
                f"Skipping an event with no UID from {self._config.name}: "
                f"summary={event.summary!r}",
                level="warning",
            )
            return None

        title, venue, city, category = self._split_summary(event.summary)

        return EventCandidate(
            id=f"{self._config.name}:{event.uid}",
            source=self._config.name,
            source_type=self._config.source_type,
            title=title,
            description=self._build_description(event.description, category),
            venue=venue,
            location=city,
            url=event.url,
            start_time=event.dtstart,
            end_time=event.dtend,
            # Deliberately unset. CREATED tracks when the calendar was last
            # rebuilt, not when the event was announced, and this field is what
            # the ingestion lookback discards on.
            raw_published_at=None,
            discovered_at=self._get_now(),
        )

    def _split_summary(
        self, summary: str | None
    ) -> tuple[str | None, str | None, str | None, str | None]:
        """Split `[Venue, City, Category] Title` into its parts.

        A summary that does not follow the convention keeps its whole text as the
        title. Venue attribution is worth losing; the event is not.
        """
        if not summary:
            return None, None, None, None

        cleaned = _INVISIBLE_RE.sub("", summary).strip()
        match = _PREFIX_RE.match(cleaned)
        if match is None:
            self._log(
                f"Summary from {self._config.name} does not follow the "
                f"[Venue, City] convention, keeping it as the title: {cleaned!r}",
                level="warning",
            )
            return cleaned, None, None, None

        parts = [p.strip() for p in match.group("prefix").split(",")]
        title = match.group("title").strip() or None
        venue = parts[0] or None if parts else None
        city = parts[1] or None if len(parts) > 1 else None
        category = parts[2] or None if len(parts) > 2 else None

        return title, venue, city, category

    @staticmethod
    def _build_description(description: str | None, category: str | None) -> str | None:
        """Combine the feed's description with the category the title declared.

        Category never becomes a tag: populated tags are the documented bypass for
        LLM Pass 1, so writing one there would suppress extraction entirely.
        """
        text = html_to_text(description) if description else None

        if category and text:
            return f"Category: {category}\n\n{text}"
        if category:
            return f"Category: {category}"
        return text

    def _log(self, message: str, level: str = "info") -> None:
        if self._logger is None:
            return
        getattr(self._logger, level)(message, component="ics_source", duration_ms=0)
