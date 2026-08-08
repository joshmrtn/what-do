"""Parser for schema.org events published as JSON-LD.

The richest structured data a site can offer and the least work to read: a
`<script type="application/ld+json">` block holding `Event` objects with real
ISO 8601 timestamps, a location, and an explicit `eventStatus`. That last field
matters — a cancellation is *stated* here, so unlike a feed whose titles carry
`*** CANCELED***` markers, nothing has to be guessed from prose.

Robustness is the whole difficulty. Sites publish several blocks per page, most
of them not events at all, and some publish blocks that are not valid JSON:
Rockport Music emits `"startDate": ""` followed by an unquoted
`"endDate": Friday, April 2, 7:30 pm`. One bad block must never cost the others.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any

#: The blocks themselves. Parsed with a regex rather than the HTML parser
#: because their content is JSON, not markup, and must not be entity-decoded.
_BLOCK_RE = re.compile(
    r'<script[^>]*type=["\']application/ld\+json["\'][^>]*>(.*?)</script>',
    re.IGNORECASE | re.DOTALL,
)

#: Statuses that mean the event is not going ahead as listed.
_DEAD_STATUSES = ("cancelled", "canceled", "postponed")


@dataclass(frozen=True)
class JsonLdEvent:
    """One `schema.org/Event`, as the markup states it."""

    title: str
    start: datetime
    end: datetime | None = None
    url: str | None = None
    venue: str | None = None
    address: str | None = None
    description: str | None = None


def _parse_moment(value: Any) -> datetime | None:
    """Read an ISO 8601 timestamp, returning None for anything unreadable."""
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        return datetime.fromisoformat(value.strip())
    except ValueError:
        return None


def _walk(node: Any) -> list[dict[str, Any]]:
    """Collect every `Event` object anywhere inside a decoded block.

    Sites nest them differently — bare arrays, `itemListElement`, an `ItemList`
    under `mainEntity` — and the shape is not worth predicting when a walk finds
    all of them.
    """
    found: list[dict[str, Any]] = []

    if isinstance(node, list):
        for item in node:
            found.extend(_walk(item))
        return found

    if not isinstance(node, dict):
        return found

    types = node.get("@type")
    names = types if isinstance(types, list) else [types]
    if any(isinstance(name, str) and name.endswith("Event") for name in names):
        found.append(node)

    for value in node.values():
        if isinstance(value, (list, dict)):
            found.extend(_walk(value))

    return found


def _text(value: Any) -> str | None:
    return value.strip() or None if isinstance(value, str) else None


def _location_of(node: Any) -> tuple[str | None, str | None]:
    """The venue name and street address, when the event names a place."""
    if not isinstance(node, dict):
        return None, None

    address = node.get("address")
    if isinstance(address, dict):
        street = _text(address.get("streetAddress"))
    else:
        street = _text(address)

    return _text(node.get("name")), street


def _is_dead(node: dict[str, Any]) -> bool:
    """Whether the site says this event is cancelled or postponed."""
    status = node.get("eventStatus")
    text = status.get("@id", "") if isinstance(status, dict) else status
    if not isinstance(text, str):
        return False
    return any(dead in text.lower() for dead in _DEAD_STATUSES)


def parse_jsonld_events(html: str, *, logger: Any = None) -> list[JsonLdEvent]:
    """Read every schema.org Event a page publishes as JSON-LD.

    Args:
        html: The page body.
        logger: Structured logger. Optional.

    Returns:
        The events in document order. An event is dropped when it has no name,
        no readable start, or a status saying it is not happening.
    """
    events: list[JsonLdEvent] = []
    unreadable = 0

    for block in _BLOCK_RE.findall(html):
        try:
            decoded = json.loads(block)
        except json.JSONDecodeError:
            # Some sites emit invalid JSON-LD. One bad block is not the page.
            unreadable += 1
            continue

        for node in _walk(decoded):
            title = _text(node.get("name"))
            start = _parse_moment(node.get("startDate"))
            if title is None or start is None or _is_dead(node):
                continue

            venue, address = _location_of(node.get("location"))
            events.append(
                JsonLdEvent(
                    title=title,
                    start=start,
                    end=_parse_moment(node.get("endDate")),
                    url=_text(node.get("url")),
                    venue=venue,
                    address=address,
                    description=_text(node.get("description")),
                )
            )

    if unreadable and logger is not None:
        logger.info(
            f"json-ld: skipped {unreadable} block(s) that were not valid JSON",
            component="jsonld",
            duration_ms=0,
        )

    return events
