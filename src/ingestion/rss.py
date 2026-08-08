"""Parser for RSS 2.0 feeds.

Written for the Squarespace `/{collection}?format=rss` pattern, which is how
several venues in this region publish events without running a calendar at all.
It reads a feed into items and stops there: an RSS item has no event date field,
so *where* the date lives is a per-site convention that only a site's own adapter
can know.

XML is parsed with the standard library. `xml.etree` is not hardened against a
hostile document, which is acceptable here only because feeds are configured by
hand rather than discovered — if that ever changes, this is the place that needs
`defusedxml`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from email.utils import parsedate_to_datetime
from xml.etree import ElementTree

#: `content:encoded`, the fuller body some feeds carry alongside `description`.
_CONTENT_ENCODED = "{http://purl.org/rss/1.0/modules/content/}encoded"


@dataclass(frozen=True)
class RssItem:
    """One `<item>`, as the feed states it."""

    title: str
    link: str | None = None
    description: str | None = None
    #: `content:encoded` where the feed carries it, otherwise None.
    content: str | None = None
    #: When the item was *posted*. Never an event date — feeds routinely
    #: announce a show weeks before it happens.
    published_at: datetime | None = None
    guid: str | None = None
    categories: list[str] = field(default_factory=list)


def _text_of(element: ElementTree.Element | None) -> str | None:
    """The element's text, stripped, or None if it has none."""
    if element is None or element.text is None:
        return None
    text = element.text.strip()
    return text or None


def _moment_of(value: str | None) -> datetime | None:
    """Read an RFC 822 date, returning None for anything unreadable."""
    if not value:
        return None
    try:
        return parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None


def parse_rss(xml: str) -> list[RssItem]:
    """Parse an RSS 2.0 document into its items.

    Args:
        xml: The feed body.

    Returns:
        One RssItem per `<item>`, in feed order. An item without a title is
        dropped — nothing downstream can identify or extract from one.

    Raises:
        ValueError: If the document is not parseable XML, or carries no channel.
            A source answering with an HTML error page is a real failure, and
            ingestion already treats a raising source as one.
    """
    try:
        root = ElementTree.fromstring(xml)
    except ElementTree.ParseError as error:
        raise ValueError(f"not a parseable RSS document: {error}") from error

    channel = root.find("channel")
    if channel is None:
        raise ValueError("RSS document has no <channel>")

    items: list[RssItem] = []
    for element in channel.findall("item"):
        title = _text_of(element.find("title"))
        if not title:
            continue

        link = _text_of(element.find("link"))
        items.append(
            RssItem(
                title=title,
                link=link,
                description=_text_of(element.find("description")),
                content=_text_of(element.find(_CONTENT_ENCODED)),
                published_at=_moment_of(_text_of(element.find("pubDate"))),
                # A feed omitting guid still has to be deduplicable, and the
                # link is the only other thing that identifies an item.
                guid=_text_of(element.find("guid")) or link,
                categories=[
                    text
                    for text in (_text_of(category) for category in element.findall("category"))
                    if text
                ],
            )
        )

    return items
