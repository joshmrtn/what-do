"""RRULE expansion for calendar sources.

Pure: a VEvent and a window in, occurrence start times out. No I/O and no clock
of its own.

Expansion has to happen *before* any filter on event time, and be bounded *by*
it. A weekly event running since 2023 carries a base occurrence two years back,
so filtering on that date deletes a live event precisely because it is
long-running. Bounding the expansion is also what makes unbounded rules
terminate — the feed this was built against has nine with neither UNTIL nor
COUNT.

The recurrence arithmetic is `dateutil`'s, which implements the whole RFC 5545
RRULE grammar. What stays here is ICS semantics: exclusions carry their own
timezone, and an unknown construct must degrade to the base occurrence rather
than vanish.
"""

from __future__ import annotations

from datetime import datetime, timezone, tzinfo
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from dateutil.rrule import rrulestr

from src.ingestion.ics import VEvent, parse_timestamp

#: A date-only value, `YYYYMMDD`, as opposed to a full DATE-TIME.
_DATE_ONLY = "YYYYMMDD"

#: What a malformed rule or exclusion may raise. `rrulestr` reports bad grammar
#: as ValueError, but a nonsensical INTERVAL can surface as either of the others.
_RULE_ERRORS = (ValueError, TypeError, OverflowError, KeyError)


def expand(
    event: VEvent,
    window_start: datetime,
    window_end: datetime,
    zone: tzinfo,
    logger: Any = None,
) -> list[datetime]:
    """List an event's occurrence start times inside a window.

    Every event goes through here, recurring or not, so callers never branch on
    whether an RRULE is present.

    Args:
        event: The parsed VEVENT.
        window_start: Inclusive start of the window.
        window_end: Exclusive end, so consecutive windows cannot both claim one
            occurrence.
        zone: Timezone assumed for naive values that declare no TZID of their own.
        logger: Structured logger for degraded rules. Optional.

    Returns:
        Aware start times in chronological order. Empty when the event has no
        start time, or when nothing falls inside the window.
    """
    if event.dtstart is None:
        return []

    start = _localise(event.dtstart, event.dtstart_tzid, zone)
    rule = _rule_text(event)

    if not rule:
        return [start] if window_start <= start < window_end else []

    try:
        occurrences = [
            occurrence
            for occurrence in rrulestr(_normalise_until(rule, start), dtstart=start).between(
                window_start, window_end, inc=True
            )
            if occurrence < window_end
        ]
    except _RULE_ERRORS as exc:
        # Keep the event on its base occurrence. Dropping it loses something
        # real; guessing at the rule would invent dates that were never
        # published.
        _warn(logger, f"Could not expand rule {rule!r}, using the base occurrence: {exc}")
        return [start] if window_start <= start < window_end else []

    excluded = _exclusions(event, zone, logger)

    return [occurrence for occurrence in occurrences if occurrence not in excluded]


def _normalise_until(rule: str, start: datetime) -> str:
    """Rewrite a non-conformant UNTIL into the UTC form dateutil demands.

    RFC 5545 requires UNTIL in UTC whenever DTSTART is zoned, and dateutil
    enforces that strictly. Real feeds do not comply — one writes bare dates
    such as `UNTIL=20210208` on 50 zoned rules. Left alone those raise, and the
    event would fall back to its base occurrence: harmless for a rule that has
    already expired, but a *future* rule written the same way would silently
    lose every occurrence after its first.

    A date-only UNTIL names a whole day, so it becomes that day's last second
    rather than its midnight — otherwise an evening event on the final day is
    dropped.
    """
    parts = []
    for part in rule.split(";"):
        key, sep, value = part.partition("=")
        if sep and key.strip().upper() == "UNTIL" and not value.strip().endswith("Z"):
            parsed, _ = parse_timestamp(value, {})
            if parsed is not None:
                if len(value.strip()) == len(_DATE_ONLY):
                    parsed = parsed.replace(hour=23, minute=59, second=59)
                as_utc = parsed.replace(tzinfo=start.tzinfo).astimezone(timezone.utc)
                part = f"{key}={as_utc.strftime('%Y%m%dT%H%M%SZ')}"
        parts.append(part)

    return ";".join(parts)


def _rule_text(event: VEvent) -> str:
    """The event's RRULE value, or empty when it declares none."""
    rules = event.repeated.get("RRULE")
    if not rules:
        return ""
    return rules[0].value.strip()


def _exclusions(event: VEvent, zone: tzinfo, logger: Any) -> set[datetime]:
    """Every EXDATE the event declares, as aware instants.

    Exclusions are matched by instant rather than by wall clock, so a UTC EXDATE
    still cancels the zoned occurrence it names.
    """
    excluded: set[datetime] = set()

    for entry in event.repeated.get("EXDATE", []):
        parsed, tzid = parse_timestamp(entry.value, entry.params)
        if parsed is None:
            _warn(logger, f"Ignoring an unparseable EXDATE: {entry.value!r}")
            continue
        excluded.add(_localise(parsed, tzid, zone))

    return excluded


def _localise(value: datetime, tzid: str | None, zone: tzinfo) -> datetime:
    """Attach a timezone to a naive value, preferring the one it declared.

    `VEvent` deliberately leaves zoned values naive and reports the zone
    alongside, so this is where the guess becomes explicit.
    """
    if value.tzinfo is not None:
        return value

    if tzid:
        try:
            return value.replace(tzinfo=ZoneInfo(tzid))
        except (ZoneInfoNotFoundError, ValueError):
            pass

    return value.replace(tzinfo=zone)


def _warn(logger: Any, message: str) -> None:
    if logger is not None:
        logger.warning(message, component="recurrence", duration_ms=0)
