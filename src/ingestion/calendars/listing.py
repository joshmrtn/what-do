"""Parser for date-grouped HTML event listings.

Aggregator pages commonly render a flat run of paragraphs: a date heading, then
category headings, then one line per event as `TIME - TITLE - VENUE - CITY`.
This walks that structure in document order, carrying the current date and
category forward onto the events beneath them.

Two rules govern the edges. An event that cannot be placed on a date is dropped,
because a guessed date is worse than a missing event. Everything else degrades:
a line missing its venue keeps its title and time.
"""

from __future__ import annotations

import html as _html
import re
from dataclasses import dataclass
from datetime import date, datetime
from html.parser import HTMLParser
from typing import Any

#: `7:00 PM - Rest of line`, tolerating the missing space a linked title leaves.
_EVENT_RE = re.compile(r"^(\d{1,2}):(\d{2})\s*(AM|PM)\s*-\s*(?P<rest>.+)$", re.IGNORECASE)

#: `Wednesday, August 5` — a weekday, a month name, and a day, with no year.
_DATE_RE = re.compile(
    r"^(?P<weekday>[A-Za-z]+day)\s*,\s*(?P<month>[A-Za-z]+)\s+(?P<day>\d{1,2})\s*$"
)

#: How far back a listing date may fall before it is read as next year instead.
_BACKWARD_TOLERANCE_DAYS = 30


@dataclass(frozen=True)
class ListingEntry:
    """One event line, with the headings that governed it applied."""

    title: str
    venue: str | None
    city: str | None
    category: str | None
    start: datetime
    url: str | None


class _ParagraphCollector(HTMLParser):
    """Collects each paragraph's text, its first link, and whether it is a heading."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.paragraphs: list[tuple[str, str | None, bool]] = []
        self._depth = 0
        self._text: list[str] = []
        self._href: str | None = None
        self._in_strong = False
        self._saw_strong_text = False
        self._saw_plain_text = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "p":
            self._depth += 1
            self._text, self._href = [], None
            self._in_strong = False
            self._saw_strong_text = False
            self._saw_plain_text = False
        elif tag == "strong" and self._depth:
            self._in_strong = True
        elif tag == "a" and self._depth and self._href is None:
            self._href = dict(attrs).get("href")

    def handle_endtag(self, tag: str) -> None:
        if tag == "p" and self._depth:
            self._depth -= 1
            text = "".join(self._text).strip()
            if text:
                # A heading is emphasised text and nothing else; an event line
                # may contain a <strong> without being one.
                is_heading = self._saw_strong_text and not self._saw_plain_text
                self.paragraphs.append((text, self._href, is_heading))
        elif tag == "strong":
            self._in_strong = False

    def handle_data(self, data: str) -> None:
        if not self._depth:
            return
        self._text.append(data)
        if not data.strip():
            return
        if self._in_strong:
            self._saw_strong_text = True
        else:
            self._saw_plain_text = True


def _resolve_year(month: int, day: int, weekday: str, today: date) -> date | None:
    """Choose the year a bare `Weekday, Month Day` heading refers to.

    The weekday name is a checksum: only one nearby year usually matches it. Ties
    are broken forward, because a listing page advertises what is coming up.
    """
    wanted = weekday.strip().lower()
    candidates: list[date] = []

    for year in (today.year, today.year + 1, today.year - 1):
        try:
            candidate = date(year, month, day)
        except ValueError:
            continue
        if candidate.strftime("%A").lower() != wanted:
            continue
        if (today - candidate).days > _BACKWARD_TOLERANCE_DAYS:
            continue
        candidates.append(candidate)

    if not candidates:
        return None

    return min(candidates, key=lambda d: abs((d - today).days))


def _parse_date_heading(text: str, today: date) -> date | None:
    """Parse `Wednesday, August 5` into a real date, or None if it is not one."""
    match = _DATE_RE.match(text.strip())
    if match is None:
        return None

    try:
        month = datetime.strptime(match.group("month")[:3], "%b").month
    except ValueError:
        return None

    return _resolve_year(month, int(match.group("day")), match.group("weekday"), today)


def _split_event(rest: str) -> tuple[str, str | None, str | None]:
    """Split `TITLE - VENUE - CITY`, treating the last two segments as location.

    Titles contain dashes often enough that counting from the left loses venues;
    counting from the right does not, since the trailing shape is fixed.
    """
    parts = [p.strip() for p in rest.split(" - ") if p.strip()]

    if len(parts) >= 3:
        return " - ".join(parts[:-2]), parts[-2], parts[-1]
    if len(parts) == 2:
        return parts[0], parts[1], None
    return rest.strip(), None, None


def parse_listing(html: str, today: date, logger: Any = None) -> list[ListingEntry]:
    """Parse a date-grouped listing page into its events.

    Args:
        html: The listing page markup.
        today: Reference date used to resolve headings that omit the year.
        logger: Structured logger for skipped lines. Optional.

    Returns:
        One entry per placeable event line, in document order.
    """
    collector = _ParagraphCollector()
    collector.feed(html)

    entries: list[ListingEntry] = []
    current_date: date | None = None
    current_category: str | None = None

    for text, href, is_heading in collector.paragraphs:
        if is_heading:
            heading_date = _parse_date_heading(text, today)
            if heading_date is not None:
                current_date, current_category = heading_date, None
            elif _DATE_RE.match(text.strip()):
                # Shaped like a date but unresolvable; drop the day rather than
                # hang its events off the previous one.
                current_date, current_category = None, None
            else:
                current_category = text.strip()
            continue

        match = _EVENT_RE.match(_html.unescape(text).strip())
        if match is None:
            continue

        if current_date is None:
            if logger is not None:
                logger.warning(
                    f"Skipping a listing line with no date heading above it: {text!r}",
                    component="listing",
                    duration_ms=0,
                )
            continue

        hour = int(match.group(1)) % 12
        if match.group(3).upper() == "PM":
            hour += 12

        title, venue, city = _split_event(match.group("rest"))
        entries.append(
            ListingEntry(
                title=title,
                venue=venue,
                city=city,
                category=current_category,
                start=datetime(
                    current_date.year,
                    current_date.month,
                    current_date.day,
                    hour,
                    int(match.group(2)),
                ),
                url=href,
            )
        )

    return entries
