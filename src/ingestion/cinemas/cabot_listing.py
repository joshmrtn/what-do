"""Parser for The Cabot's `whats-on` listing.

Pure: HTML and a reference date in, events out. No network and no clock.

The Cabot is a restored theatre rather than only a cinema, so its listing mixes
touring music, comedy, talks and workshops with `$1 Movie Series` screenings.
Its WordPress REST API exposes a `cabot_events` post type but **not the event
date** — only the post's own creation date, with the real one buried in prose in
an SEO description. The listing markup carries it properly, so that is what this
reads.

Dates carry neither a year nor a weekday, so unlike the other listing parsers
there is no checksum available. The listing is strictly ascending — measured
across all nine pages, 7 August to 21 December — so the year is carried forward
and rolled whenever a date would otherwise go backwards.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, datetime, time
from html.parser import HTMLParser
from typing import Any

#: `event_item_20591` — the site's own post id.
_ID_RE = re.compile(r"event_item_(?P<id>\d+)")

#: `8:00pm`, or `10:30am`.
_TIME_RE = re.compile(r"(?P<hour>\d{1,2}):(?P<minute>\d{2})\s*(?P<meridiem>am|pm)", re.I)

#: `7 Aug 8:00pm`, or `11 - 25 Aug` for a run, as the date block reads once
#: its nested time div is flattened into it.
_DATE_RE = re.compile(
    r"(?P<from>\d{1,2})\s*(?:-\s*(?P<to>\d{1,2}))?\s+(?P<month>[A-Za-z]{3,})"
)

#: `Showing 1-10 of 88 events`.
_TOTAL_RE = re.compile(r"of\s+(?P<total>\d+)\s+events", re.I)

#: How far back a listed date may fall before it is read as next year instead.
_BACKWARD_TOLERANCE_DAYS = 30


@dataclass(frozen=True)
class CabotEvent:
    """One listed event.

    Attributes:
        event_id: The site's own post id, stable across runs.
        title: Event title.
        start: Local wall-clock start. Naive — the caller localises.
        end: Last day of a run, for listings such as `11 - 25 Aug`. None for a
            single date.
        time_known: False when the listing gave a date but no time, so a caller
            can avoid presenting midnight as though it were a start time.
        url: Link to the event's own page.
        genres: Labels the listing applied, e.g. Music, Comedy.
        subtitle: The listing's secondary line. Overloaded: a venue and address
            when `off_site` is set, otherwise a tour or series name such as
            `INDIGO PARK TOUR`.
        off_site: Whether the listing flagged this as happening away from the
            main theatre. Read from an explicit marker rather than guessed from
            the subtitle's text, which carries both meanings.
    """

    event_id: str
    title: str
    start: datetime
    end: datetime | None = None
    time_known: bool = True
    url: str | None = None
    genres: list[str] = field(default_factory=list)
    subtitle: str | None = None
    off_site: bool = False


@dataclass
class _Raw:
    """One event block as read, before its year is decided."""

    event_id: str = ""
    date_text: str = ""
    title: str = ""
    url: str | None = None
    genres: list[str] = field(default_factory=list)
    subtitle: str | None = None
    off_site: bool = False


class _ListingCollector(HTMLParser):
    """Walks `event_item` blocks, collecting the fields each one declares.

    The date block nests a `time` div inside itself, so it is captured whole and
    read with a regex rather than element by element — switching capture
    mid-block silently dropped the month.
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.items: list[_Raw] = []
        self.total: int | None = None
        self._current: _Raw | None = None
        self._capture: str | None = None
        self._capture_tag: str | None = None
        self._depth = 0
        self._text: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        classes = (attributes.get("class") or "").split()

        if self._capture is not None:
            # Inside a captured block. Only the opening tag's own type nests —
            # the date block holds a span and a div, and closing on the first
            # end tag of any kind truncated it to the day number.
            if tag == self._capture_tag:
                self._depth += 1
            return

        if "event_item" in classes:
            self._flush()
            match = _ID_RE.search(attributes.get("id") or "")
            self._current = _Raw(event_id=match.group("id") if match else "")
        elif "results_count" in classes:
            self._begin("total", tag)
        elif self._current is None:
            return
        elif "event_date" in classes:
            self._begin("date", tag)
        elif "genre" in classes:
            self._begin("genre", tag)
        elif "h4" in classes:
            self._begin("title", tag)
        elif "h5" in classes:
            self._begin("subtitle", tag)
        elif "off_cabot_logo" in classes:
            self._current.off_site = True
        elif tag == "a" and self._current.url is None:
            href = attributes.get("href")
            if href and "/event/" in href:
                self._current.url = href

    def handle_endtag(self, tag: str) -> None:
        if self._capture is None or tag != self._capture_tag:
            return

        self._depth -= 1
        if self._depth > 0:
            return

        text = " ".join("".join(self._text).split())
        what, self._capture, self._capture_tag = self._capture, None, None

        if what == "total":
            match = _TOTAL_RE.search(text)
            self.total = int(match.group("total")) if match else None
            return

        if self._current is None:
            return

        if what == "date":
            self._current.date_text = text
        elif what == "genre":
            if text:
                self._current.genres.append(text)
        elif what == "title":
            self._current.title = text
        elif what == "subtitle":
            self._current.subtitle = text or None

    def handle_data(self, data: str) -> None:
        if self._capture is not None:
            self._text.append(data)

    def close(self) -> None:
        super().close()
        self._flush()

    def _begin(self, what: str, tag: str) -> None:
        self._capture, self._capture_tag = what, tag
        self._text, self._depth = [], 1

    def _flush(self) -> None:
        if self._current is not None:
            self.items.append(self._current)
            self._current = None


def parse_cabot(
    html: str, today: date, logger: Any = None, want_total: bool = False
) -> Any:
    """Parse a Cabot listing page into its events.

    Args:
        html: The fetched page.
        today: Reference date, used to resolve the missing year.
        logger: Structured logger for skipped rows. Optional.
        want_total: Also return the `Showing N of M events` total, for a caller
            deciding how many pages to fetch.

    Returns:
        The events in page order, or `(events, total)` when `want_total` is set.
        A row whose date or title cannot be read is skipped rather than guessed.
    """
    collector = _ListingCollector()
    collector.feed(html)
    collector.close()

    events: list[CabotEvent] = []
    year = today.year
    previous: date | None = None

    for raw in collector.items:
        if not raw.title:
            _warn(logger, f"Skipping a listing row with no title: id={raw.event_id!r}")
            continue

        parts = _DATE_RE.search(raw.date_text)
        month = _month_of(parts.group("month")) if parts else None
        if parts is None or month is None:
            _warn(logger, f"Skipping {raw.title!r}: unreadable date {raw.date_text!r}")
            continue
        days = parts

        start_day, year, previous = _place(
            int(days.group("from")), month, year, previous, today
        )
        if start_day is None:
            continue

        moment = _time_of(raw.date_text)
        end_day = None
        if days.group("to"):
            end_day = _end_of_run(start_day, int(days.group("to")))

        events.append(
            CabotEvent(
                event_id=raw.event_id,
                title=raw.title,
                start=datetime.combine(start_day, moment or time(0, 0)),
                end=datetime.combine(end_day, time(23, 59)) if end_day else None,
                time_known=moment is not None,
                url=raw.url,
                genres=raw.genres,
                subtitle=raw.subtitle,
                off_site=raw.off_site,
            )
        )

    return (events, collector.total) if want_total else events


def _place(
    day: int, month: int, year: int, previous: date | None, today: date
) -> tuple[date | None, int, date | None]:
    """Give a bare day and month the year the listing's ordering implies."""
    for attempt in (year, year + 1):
        try:
            candidate = date(attempt, month, day)
        except ValueError:
            return None, year, previous

        if previous is not None and candidate < previous:
            continue
        if previous is None and (today - candidate).days > _BACKWARD_TOLERANCE_DAYS:
            continue
        return candidate, attempt, candidate

    return None, year, previous


def _end_of_run(start: date, last_day: int) -> date | None:
    """The closing date of a run such as `11 - 25 Aug`, which shares its month."""
    try:
        return start.replace(day=last_day)
    except ValueError:
        return None


def _month_of(text: str) -> int | None:
    try:
        return datetime.strptime(text[:3], "%b").month
    except (ValueError, IndexError):
        return None


def _time_of(text: str) -> time | None:
    """Read the time out of a date block, which may not carry one."""
    match = _TIME_RE.search(text)
    if match is None:
        return None

    hour = int(match.group("hour")) % 12
    if match.group("meridiem").lower() == "pm":
        hour += 12

    return time(hour, int(match.group("minute")))


def _warn(logger: Any, message: str) -> None:
    if logger is not None:
        logger.warning(message, component="cabot", duration_ms=0)
