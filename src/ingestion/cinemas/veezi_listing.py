"""Parser for a Veezi public ticketing `sessions` page.

Pure: HTML and a reference date in, showings out. No network and no clock.

Veezi hosts a public sessions page per cinema, keyed by a `siteToken` that
appears in the cinema's own booking links. It needs no credentials — unlike the
Veezi *API*, whose key is issued from the exhibitor's back office and is not
obtainable by a member of the public.

Two things about the markup shape the parser. The page lists each showing more
than once — 60 of 144 rows on a measured page — so showings are deduplicated on
the session id in the booking link, which is the source's own key and was
measured 1:1 against distinct showings with no collisions. And day headings omit
the year, so the weekday name is used as a checksum.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime, time
from html.parser import HTMLParser
from typing import Any

from src.ingestion.calendars.listing import resolve_year

#: `Friday 7, August` — weekday, day, month, and no year.
_DATE_RE = re.compile(
    r"^(?P<weekday>[A-Za-z]+day)\s+(?P<day>\d{1,2})\s*,\s*(?P<month>[A-Za-z]+)\s*$"
)

#: `7:00 PM`, as the page writes every showtime.
_TIME_RE = re.compile(r"^(?P<hour>\d{1,2}):(?P<minute>\d{2})\s*(?P<meridiem>AM|PM)$", re.I)

#: The session id in a booking link: `/purchase/38750?siteToken=...`.
_SESSION_RE = re.compile(r"/purchase/(?P<id>\d+)")


@dataclass(frozen=True)
class VeeziSession:
    """One showing of one film.

    Attributes:
        session_id: The cinema's own id for this showing, from its booking link.
        title: Film title.
        start: Local wall-clock start. Naive — the caller localises, since only
            it knows the cinema's zone.
        url: Booking link for this showing.
    """

    session_id: str
    title: str
    start: datetime
    url: str


class _SessionCollector(HTMLParser):
    """Walks film blocks, tracking the title and date each showtime sits under."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.rows: list[tuple[str, str, str, str]] = []
        self._title: str | None = None
        self._heading: str | None = None
        self._href: str | None = None
        self._capture: str | None = None
        self._text: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        classes = (attributes.get("class") or "").split()

        if tag == "div" and "film" in classes:
            # A new film block: the previous title must not leak into it.
            self._title = None
        elif tag == "h3" and "title" in classes:
            self._capture, self._text = "title", []
        elif tag == "h4" and "date" in classes:
            self._capture, self._text = "date", []
        elif tag == "a":
            self._href = attributes.get("href")
        elif tag == "time":
            self._capture, self._text = "time", []

    def handle_endtag(self, tag: str) -> None:
        if self._capture is None:
            if tag == "a":
                self._href = None
            return

        text = " ".join("".join(self._text).split())

        if tag == "h3" and self._capture == "title":
            self._title = text or None
        elif tag == "h4" and self._capture == "date":
            self._heading = text or None
        elif tag == "time" and self._capture == "time":
            if self._title and self._heading and self._href:
                self.rows.append((self._href, self._title, self._heading, text))

        self._capture = None

    def handle_data(self, data: str) -> None:
        if self._capture is not None:
            self._text.append(data)


def parse_sessions(html: str, today: date, logger: Any = None) -> list[VeeziSession]:
    """Parse a Veezi sessions page into its showings.

    Args:
        html: The fetched page.
        today: Reference date used to resolve headings that omit the year.
        logger: Structured logger for skipped rows. Optional.

    Returns:
        One VeeziSession per distinct showing, in page order. A row whose date,
        time, or booking link cannot be read is skipped rather than guessed at.
    """
    collector = _SessionCollector()
    collector.feed(html)

    sessions: list[VeeziSession] = []
    seen: set[str] = set()

    for href, title, heading, raw_time in collector.rows:
        session_id = _session_id(href)
        if session_id is None:
            _warn(logger, f"Skipping a showing with no session id: {title!r} at {raw_time!r}")
            continue
        if session_id in seen:
            # The page repeats showings; the id is what makes the count honest.
            continue

        day = _parse_heading(heading, today)
        if day is None:
            _warn(logger, f"Skipping a showing under an unreadable date: {heading!r}")
            continue

        moment = _parse_time(raw_time)
        if moment is None:
            _warn(logger, f"Skipping a showing with an unreadable time: {raw_time!r}")
            continue

        seen.add(session_id)
        sessions.append(
            VeeziSession(
                session_id=session_id,
                title=title,
                start=datetime.combine(day, moment),
                url=href,
            )
        )

    return sessions


def _session_id(href: str) -> str | None:
    match = _SESSION_RE.search(href)

    return match.group("id") if match else None


def _parse_heading(heading: str, today: date) -> date | None:
    """Resolve `Friday 7, August` against the year its weekday agrees with."""
    match = _DATE_RE.match(heading)
    if match is None:
        return None

    try:
        month = datetime.strptime(match.group("month")[:3], "%b").month
    except ValueError:
        return None

    return resolve_year(month, int(match.group("day")), match.group("weekday"), today)


def _parse_time(raw: str) -> time | None:
    """Parse `7:00 PM`, keeping noon and midnight distinct."""
    match = _TIME_RE.match(raw.strip())
    if match is None:
        return None

    hour = int(match.group("hour")) % 12
    if match.group("meridiem").upper() == "PM":
        hour += 12

    minute = int(match.group("minute"))
    if not 0 <= minute < 60:
        return None

    return time(hour, minute)


def _warn(logger: Any, message: str) -> None:
    if logger is not None:
        logger.warning(message, component="veezi", duration_ms=0)
