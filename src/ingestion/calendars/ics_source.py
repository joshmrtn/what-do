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
from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
from typing import Any, Callable

import requests

from src.storage.protocols import HttpCache
from src.config import DEFAULT_DAY_STARTS_AT, DEFAULT_HORIZON_DAYS, FeedConfig
from src.ingestion.calendars.fetching import fetch_document
from src.ingestion.ics import VEvent, parse_ics
from src.ingestion.recurrence import Occurrence, expand_calendar
from src.ingestion.source import IngestionSource
from src.utils.nights import night_start
from src.models.event_candidate import EventCandidate
from src.models.timing import ALL_DAY, EXACT
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
        http_cache: HttpCache,
        session: requests.Session | None = None,
        get_now: Callable[[], datetime] = datetime.now,
        logger: Any = None,
        timezone_name: str = "UTC",
        horizon_days: int = DEFAULT_HORIZON_DAYS,
        day_starts_at: time = DEFAULT_DAY_STARTS_AT,
    ) -> None:
        self._config = config
        self._http_cache = http_cache
        self._session = session or requests.Session()
        self._get_now = get_now
        self._logger = logger
        self._zone = _zone_of(timezone_name)
        self._horizon_days = horizon_days
        self._day_starts_at = day_starts_at
        #: Summaries in the current fetch that ignored the [Venue, City]
        #: convention, reported once at the end rather than one line each.
        self._unconventional = 0

    @property
    def source_name(self) -> str:
        """The feed's configured name, so a report can name the feed not the class."""
        return self._config.name

    def fetch(self) -> list[EventCandidate]:
        """Fetch and parse the calendar, skipping the network when it is polite to.

        Recurring events are expanded into one candidate per occurrence inside
        the horizon. Without that a weekly event running since 2023 arrives as a
        single candidate dated 2023, and every live occurrence is lost the
        moment anything filters on event time.

        Returns:
            One EventCandidate per occurrence, in chronological order.
        """
        events = self._read_events()

        # Floored at the night, not the instant: an event under way has not
        # stopped being tonight's just because the job looking at it started
        # afterwards.
        window_start = night_start(self._get_now(), self._day_starts_at, self._zone)
        window_end = window_start + timedelta(days=self._horizon_days)
        occurrences = expand_calendar(
            events, window_start, window_end, self._zone, logger=self._logger
        )

        self._unconventional = 0
        candidates = []
        for occurrence in occurrences:
            candidate = self._to_candidate(occurrence)
            if candidate is not None:
                candidates.append(candidate)

        self._report_unconventional(len(occurrences))

        return candidates

    def _report_unconventional(self, total: int) -> None:
        """Report summaries that ignored the convention, once rather than each.

        Silent when the feed declares its own venue: a cinema's calendar names
        the film, and nothing is missing once config has said where it plays.
        Otherwise one line — a feed measured at 12,038 such events would
        otherwise bury every other line in the run.
        """
        if not self._unconventional or self._config.venue:
            return

        self._log(
            f"{self._unconventional} of {total} summaries from "
            f"{self._config.name} did not follow the [Venue, City] convention, "
            "so they have no venue. Set `venue:` on the feed if it has only one.",
            level="warning",
        )

    # ------------------------------------------------------------------
    # Fetching
    # ------------------------------------------------------------------

    def _read_events(self) -> list[VEvent]:
        """Every VEVENT this source can reach, deduplicated by identity.

        One document for an ordinary feed. A subclass may return several — the
        events are merged here rather than at the parser, because two documents
        from the same calendar overlap and a duplicated master would expand into
        duplicated occurrences.
        """
        merged: dict[tuple[str | None, str | None], VEvent] = {}
        for body in self._read_documents():
            for event in parse_ics(body, logger=self._logger):
                recurrence = event.repeated.get("RECURRENCE-ID")
                key = (event.uid, recurrence[0].value if recurrence else None)
                merged.setdefault(key, event)

        return list(merged.values())

    def _read_documents(self) -> list[str]:
        """The documents making up this calendar. One, unless a subclass says otherwise."""
        return [self._read_feed(self._config.url)]

    def _read_feed(self, url: str) -> str:
        """Return a feed body, from cache when refetching would be impolite."""
        return fetch_document(
            url,
            session=self._session,
            http_cache=self._http_cache,
            get_now=self._get_now,
            min_fetch_interval_hours=self._config.min_fetch_interval_hours,
            label=self._config.name,
            logger=self._logger,
        )

    # ------------------------------------------------------------------
    # Mapping
    # ------------------------------------------------------------------

    def _to_candidate(self, occurrence: Occurrence) -> EventCandidate | None:
        """Map one occurrence, or None when it cannot be used."""
        event = occurrence.event

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
            id=self._candidate_id(occurrence, event.uid),
            source=self._config.name,
            source_type=self._config.source_type,
            title=title,
            description=self._build_description(event.description, category),
            # A summary that names a venue is more specific than the feed's
            # blanket attribution, so the configured default only fills a gap.
            venue=venue or self._config.venue,
            location=city or self._config.city,
            url=event.url,
            start_time=_place_start(occurrence.start, event.dtstart_is_date, self._day_starts_at),
            end_time=self._end_of(occurrence),
            timing=ALL_DAY if event.dtstart_is_date else EXACT,
            # Deliberately unset. CREATED tracks when the calendar was last
            # rebuilt, not when the event was announced, and this field is what
            # the ingestion lookback discards on.
            raw_published_at=None,
            discovered_at=self._get_now(),
        )

    def _candidate_id(self, occurrence: Occurrence, uid: str) -> str:
        """Derive a stable id, adding the slot only when the UID is ambiguous.

        Every occurrence of a series shares one UID, so a recurring event needs
        its slot to tell tonight's showing from next week's. A one-off has
        nothing to disambiguate, and appending its start would churn the id the
        moment the feed corrected a time — so it keeps the bare UID.

        The slot is the *original* one, not where a moved instance landed, so
        rescheduling updates the stored event instead of orphaning it.
        """
        base = f"{self._config.name}:{uid}"
        if not occurrence.from_series:
            return base

        return f"{base}:{occurrence.original_start.strftime('%Y%m%dT%H%M%S%z')}"

    def _end_of(self, occurrence: Occurrence) -> datetime | None:
        """The occurrence's end, holding the series' duration against its start."""
        event = occurrence.event
        if event.dtend is None or event.dtstart is None:
            return None

        return occurrence.start + (event.dtend - event.dtstart)

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
            self._unconventional += 1
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


def _zone_of(name: str) -> ZoneInfo:
    """Resolve a zone name, falling back to UTC rather than failing the fetch.

    Only reached for naive feed values that declare no TZID of their own, so a
    fallback shifts a handful of timestamps rather than losing the source.
    """
    try:
        return ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError):
        return ZoneInfo("UTC")


def _place_start(start: datetime, is_all_day: bool, day_starts_at: time) -> datetime:
    """Move an all-day start off midnight and onto its own night.

    A date-only value parses to 00:00, which falls *before* the rollover and so
    lands the event on the previous night — off by one, every time. Placing it
    at the rollover hour puts it in the night it belongs to. Nothing reads this
    as a published time, because `timing` says it is not.
    """
    if not is_all_day:
        return start

    return start.replace(hour=day_starts_at.hour, minute=day_starts_at.minute)
