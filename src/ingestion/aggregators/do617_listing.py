"""Parser for Do617's schema.org microdata listings.

Every event on a Do617 listing is marked up as `schema.org/Event`, which makes
this the only HTML source in the project that needs no date inference at all:
`startDate` carries a full ISO 8601 timestamp with the UTC offset already
applied. Elsewhere we reconstruct a year from context and localise a wall clock;
here the source states both.

Two structural details govern the walk. Cards nest a `schema.org/Place` inside
the event, and *both* carry `itemprop="name"` — so a name means the venue only
while inside the location scope, and the event's title otherwise. And identity
comes from `data-permalink`, the site's own canonical path, rather than the
`data-ds-id` attribute that also appears on upvote controls and so does not
count 1:1 with events.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from html.parser import HTMLParser
from typing import Any

#: Marks the container of one event.
_EVENT_TYPE = "http://schema.org/Event"

#: Elements that never close. The microdata lives on `<meta>`, which the markup
#: writes both as `<meta ... />` and as `<meta ...>` — counting either as a level
#: of nesting drifts the depth upward and a card then never appears to end.
_VOID_TAGS = frozenset(
    {
        "area", "base", "br", "col", "embed", "hr", "img", "input",
        "link", "meta", "param", "source", "track", "wbr",
    }
)


@dataclass(frozen=True)
class Do617Event:
    """One listed event, as the markup states it."""

    permalink: str
    title: str
    start: datetime
    end: datetime | None = None
    venue: str | None = None
    venue_slug: str | None = None
    street: str | None = None
    city: str | None = None
    region: str | None = None
    latitude: float | None = None
    longitude: float | None = None


@dataclass(frozen=True)
class Do617Page:
    """One listing page: its events, and where the next one lives."""

    events: list[Do617Event]
    next_page_url: str | None = None


def _parse_moment(value: str | None) -> datetime | None:
    """Read an ISO 8601 timestamp, returning None for anything unreadable."""
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.strip())
    except ValueError:
        return None


def _parse_number(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


class _CardCollector(HTMLParser):
    """Walks the document, collecting one record per `schema.org/Event`."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.cards: list[dict[str, Any]] = []
        self.next_page_url: str | None = None
        self._depth = 0
        self._card: dict[str, Any] | None = None
        self._card_depth = 0
        self._location_depth: int | None = None
        self._capturing: str | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = {name: (value or "") for name, value in attrs}
        if tag not in _VOID_TAGS:
            self._depth += 1
        self._read(tag, attributes)

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        # `<meta ... />` carries attributes but opens no scope, so it must not
        # be read as an open-then-close pair.
        self._read(tag, {name: (value or "") for name, value in attrs})

    def _read(self, tag: str, attributes: dict[str, str]) -> None:
        if self.next_page_url is None and "ds-next-page" in attributes.get("class", ""):
            self.next_page_url = attributes.get("href") or None

        if self._card is None:
            if attributes.get("itemtype") == _EVENT_TYPE:
                self._begin_card(attributes)
            return

        itemprop = attributes.get("itemprop")

        # The location's own scope, so a nested name reads as the venue.
        if itemprop == "location":
            self._location_depth = self._depth
        elif itemprop == "startDate":
            self._card["start"] = attributes.get("datetime") or attributes.get("content")
        elif itemprop == "endDate":
            self._card["end"] = attributes.get("datetime") or attributes.get("content")
        elif itemprop in ("streetAddress", "addressLocality", "addressRegion"):
            self._card[itemprop] = attributes.get("content")
        elif itemprop in ("latitude", "longitude"):
            self._card[itemprop] = attributes.get("content")
        elif itemprop == "url" and self._in_location() and tag == "a":
            self._card["venue_slug"] = _slug_of(attributes.get("href"))
        elif itemprop == "name":
            self._capturing = "venue" if self._in_location() else "title"

    def handle_endtag(self, tag: str) -> None:
        if tag in _VOID_TAGS:
            return
        if self._capturing is not None:
            self._capturing = None
        if self._location_depth is not None and self._depth <= self._location_depth:
            self._location_depth = None
        if self._card is not None and self._depth <= self._card_depth:
            self.cards.append(self._card)
            self._card = None
        self._depth = max(0, self._depth - 1)

    def handle_data(self, data: str) -> None:
        if self._card is None or self._capturing is None:
            return
        text = data.strip()
        if text:
            self._card[self._capturing] = text

    def _begin_card(self, attributes: dict[str, str]) -> None:
        self._card = {"permalink": attributes.get("data-permalink") or ""}
        self._card_depth = self._depth
        self._location_depth = None
        self._capturing = None

    def _in_location(self) -> bool:
        return self._location_depth is not None


def _slug_of(href: str | None) -> str | None:
    """Read `gulu-gulu-cafe` out of `/venues/gulu-gulu-cafe`."""
    if not href:
        return None
    slug = href.rstrip("/").rsplit("/", 1)[-1]
    return slug or None


def parse_do617(html: str, *, logger: Any = None) -> Do617Page:
    """Parse one Do617 listing page.

    Args:
        html: The page body.
        logger: Structured logger. Optional.

    Returns:
        The page's events in document order, and its next-page URL if it has one.
        An event missing a permalink or a readable start is dropped — identity and
        a date are the two things nothing downstream can reconstruct.
    """
    collector = _CardCollector()
    collector.feed(html)
    collector.close()

    events: list[Do617Event] = []
    dropped = 0

    for card in collector.cards:
        permalink = str(card.get("permalink") or "")
        start = _parse_moment(card.get("start"))
        if not permalink or start is None:
            dropped += 1
            continue

        events.append(
            Do617Event(
                permalink=permalink,
                title=str(card.get("title") or ""),
                start=start,
                end=_parse_moment(card.get("end")),
                venue=card.get("venue"),
                venue_slug=card.get("venue_slug"),
                street=card.get("streetAddress"),
                city=card.get("addressLocality"),
                region=card.get("addressRegion"),
                latitude=_parse_number(card.get("latitude")),
                longitude=_parse_number(card.get("longitude")),
            )
        )

    if dropped and logger is not None:
        logger.info(
            f"do617: dropped {dropped} card(s) with no permalink or no readable start",
            component="do617",
            duration_ms=0,
        )

    return Do617Page(events=events, next_page_url=collector.next_page_url)
