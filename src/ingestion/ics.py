"""ICS (RFC 5545) calendar parsing.

Deliberately hand-rolled rather than pulling in a dependency: public event
calendars use a narrow slice of the format, and the parts that actually bite are
small and pure — line unfolding, value unescaping, and timestamp forms.

The parser's contract is to lose nothing. Every property a VEVENT declares is
kept in `VEvent.properties`, so a source adapter can map new fields without the
parser changing, and downstream LLM extraction sees whatever the feed sent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

#: Value forms we understand, longest first so the zoned form wins.
_UTC_FORMAT = "%Y%m%dT%H%M%SZ"
_LOCAL_FORMAT = "%Y%m%dT%H%M%S"
_DATE_FORMAT = "%Y%m%d"


@dataclass(frozen=True)
class VEvent:
    """One VEVENT block, with its raw properties preserved alongside the parsed ones.

    Timestamps are timezone-aware only when the feed said UTC. A zoned or all-day
    value stays naive and reports its declared zone in `dtstart_tzid` / `dtend_tzid`,
    so a caller localises deliberately rather than inheriting a silent guess.
    """

    uid: str | None = None
    summary: str | None = None
    description: str | None = None
    location: str | None = None
    url: str | None = None
    status: str | None = None
    dtstart: datetime | None = None
    dtend: datetime | None = None
    dtstart_tzid: str | None = None
    dtend_tzid: str | None = None
    properties: dict[str, str] = field(default_factory=dict)


def _unfold(text: str) -> list[str]:
    """Join RFC 5545 continuation lines, which begin with a space or tab."""
    normalised = text.replace("\r\n", "\n").replace("\r", "\n")
    lines: list[str] = []

    for line in normalised.split("\n"):
        if line[:1] in (" ", "\t") and lines:
            lines[-1] += line[1:]
        else:
            lines.append(line)

    return lines


def _unescape(value: str) -> str:
    """Resolve the four escapes RFC 5545 defines for TEXT values."""
    out: list[str] = []
    index = 0

    while index < len(value):
        char = value[index]
        if char == "\\" and index + 1 < len(value):
            nxt = value[index + 1]
            # \n and \N are newlines; \, \; \\ are themselves, as is anything
            # else escaped without cause.
            out.append("\n" if nxt in ("n", "N") else nxt)
            index += 2
            continue
        out.append(char)
        index += 1

    return "".join(out)


def _split_property(line: str) -> tuple[str, dict[str, str], str] | None:
    """Split `NAME;PARAM=V:value` into its name, parameters, and raw value."""
    colon = line.find(":")
    if colon == -1:
        return None

    head, value = line[:colon], line[colon + 1 :]
    parts = head.split(";")
    name = parts[0].strip().upper()
    if not name:
        return None

    params: dict[str, str] = {}
    for param in parts[1:]:
        if "=" in param:
            key, val = param.split("=", 1)
            params[key.strip().upper()] = val.strip()

    return name, params, value


def _parse_timestamp(value: str, params: dict[str, str]) -> tuple[datetime | None, str | None]:
    """Parse a DATE-TIME or DATE value, reporting any declared timezone."""
    tzid = params.get("TZID")
    raw = value.strip()

    if raw.endswith("Z"):
        try:
            return datetime.strptime(raw, _UTC_FORMAT).replace(tzinfo=timezone.utc), None
        except ValueError:
            return None, tzid

    for fmt in (_LOCAL_FORMAT, _DATE_FORMAT):
        try:
            return datetime.strptime(raw, fmt), tzid
        except ValueError:
            continue

    return None, tzid


def _build_event(properties: dict[str, str], raw: dict[str, dict[str, str]]) -> VEvent:
    """Assemble a VEvent from the properties collected for one block."""
    dtstart, dtstart_tzid = (None, None)
    dtend, dtend_tzid = (None, None)

    if "DTSTART" in properties:
        dtstart, dtstart_tzid = _parse_timestamp(properties["DTSTART"], raw["DTSTART"])
    if "DTEND" in properties:
        dtend, dtend_tzid = _parse_timestamp(properties["DTEND"], raw["DTEND"])

    return VEvent(
        uid=properties.get("UID"),
        summary=properties.get("SUMMARY"),
        description=properties.get("DESCRIPTION"),
        location=properties.get("LOCATION"),
        url=properties.get("URL"),
        status=properties.get("STATUS"),
        dtstart=dtstart,
        dtend=dtend,
        dtstart_tzid=dtstart_tzid,
        dtend_tzid=dtend_tzid,
        properties=properties,
    )


def parse_ics(text: str, logger: Any = None) -> list[VEvent]:
    """Parse an ICS calendar into its VEVENT blocks.

    Nested blocks such as VALARM are skipped, and non-VEVENT components such as
    VTIMEZONE are ignored entirely.

    Args:
        text: Raw ICS document.
        logger: Structured logger for recurrence warnings. Optional.

    Returns:
        One VEvent per VEVENT block, in document order.
    """
    events: list[VEvent] = []
    properties: dict[str, str] = {}
    params: dict[str, dict[str, str]] = {}

    depth = 0  # nesting inside a VEVENT, so VALARM does not leak properties in

    for line in _unfold(text):
        parsed = _split_property(line)
        if parsed is None:
            continue
        name, line_params, value = parsed

        if name == "BEGIN":
            if value.strip().upper() == "VEVENT" and depth == 0:
                depth = 1
                properties, params = {}, {}
            elif depth:
                depth += 1
            continue

        if name == "END":
            if depth == 1 and value.strip().upper() == "VEVENT":
                events.append(_build_event(properties, params))
                depth = 0
            elif depth > 1:
                depth -= 1
            continue

        if depth != 1:
            continue

        properties[name] = _unescape(value)
        params[name] = line_params

        if name in ("RRULE", "RECURRENCE-ID") and logger is not None:
            logger.warning(
                f"Recurring event is not expanded, using its base occurrence: "
                f"{name}={value}",
                component="ics",
                duration_ms=0,
            )

    return events
