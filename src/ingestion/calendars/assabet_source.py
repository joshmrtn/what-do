"""Assabet Interactive calendar feeds, read through the RSS source.

Assabet runs a great many Massachusetts library calendars, so this adapter is
worth more than the one feed it was written for. Salem Public Library reaches it
through an embed: `salempl.org/calendar/` is a WordPress page whose body is an
`<iframe src="https://salempl.assabetinteractive.com/calendar/">`, and the feed
lives on that host, not the library's own domain.

**`pubDate` is the event start here**, which is the exact opposite of MOON, where
it is the announcement date. Two sites, one format, opposite meanings for the
same field — which is the whole reason `RssFeedSource` leaves interpretation to
its subclasses.

Every description opens with the same four structural lines before its prose:

    Saturday, August 8, 2026 10:30—11:00 AM
    Children's Program Room - Ground Floor      <- the room, or the place
    The Salem Public Library                    <- the organisation
    370 Essex St, Salem, MA, 01970              <- the organisation's address
    An adaptive sensory-friendly song and movement program.
"""

from __future__ import annotations

import re
from datetime import datetime

from src.ingestion.calendars.rss_source import RssEvent, RssFeedSource, looks_cancelled
from src.ingestion.rss import RssItem
from src.models.timing import EXACT
from src.utils.html import html_to_text

#: How many lines of preamble precede the prose.
_PREAMBLE_LINES = 4

#: Index of each preamble line that carries something we keep.
_PLACE_LINE = 1
_ORGANISATION_LINE = 2
_ADDRESS_LINE = 3

#: `Salem Farmers' Market - 32 Derby Square, Salem, MA 01970` — a place followed
#: by its street address. A room like `Children's Program Room - Ground Floor`
#: uses the same separator and must survive intact, so the tail has to look like
#: an address before it is dropped.
_PLACE_WITH_ADDRESS_RE = re.compile(r"^(?P<place>.+?)\s+-\s+(?P<address>\d.*)$")


class AssabetRssSource(RssFeedSource):
    """Reads an Assabet Interactive `upcoming-events.rss` feed."""

    def interpret(self, item: RssItem) -> RssEvent | None:
        """Read one event out of a feed item.

        Args:
            item: The parsed `<item>`.

        Returns:
            The event, or None if it carries no date or has been cancelled.
        """
        if looks_cancelled(item.title):
            return None
        if item.published_at is None:
            return None

        start = item.published_at
        if start.tzinfo is None:
            start = start.replace(tzinfo=self.zone)

        lines = [line.strip() for line in html_to_text(item.description or "").split("\n")]
        lines = [line for line in lines if line]

        return RssEvent(
            title=item.title,
            start=start,
            # pubDate carries the hour, so nothing is placed.
            timing=EXACT,
            venue=_venue_in(lines),
            description=_prose_in(lines),
        )


def _venue_in(lines: list[str]) -> str | None:
    """Where the event actually is, which is not always the organisation.

    The address line decides. When the organisation has one, the event is at the
    organisation and the line above it is a room inside it. When that line is
    empty of a street — `, Salem, MA, 01970` — the organisation is a programme
    grouping such as `Community Visits` rather than somewhere you can go, and
    the place line is the real venue.
    """
    if len(lines) <= _ADDRESS_LINE:
        return None

    address = lines[_ADDRESS_LINE].lstrip(" ,")
    if address and address[0].isdigit():
        return lines[_ORGANISATION_LINE] or None

    return _place_of(lines[_PLACE_LINE])


def _place_of(line: str) -> str | None:
    """The place name, with any street address trimmed off the end."""
    match = _PLACE_WITH_ADDRESS_RE.match(line)
    return (match.group("place") if match else line) or None


def _prose_in(lines: list[str]) -> str | None:
    """Everything after the preamble.

    The four lines it drops are date, room, organisation and address — all of
    which are already structured fields. Leaving them in would have extraction
    read a street address as part of what the event *is*.
    """
    prose = "\n".join(lines[_PREAMBLE_LINES:]).strip()
    return prose or None
