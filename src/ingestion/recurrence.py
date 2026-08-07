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

from dataclasses import dataclass
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


@dataclass(frozen=True)
class Occurrence:
    """One concrete instance of an event, and what describes it.

    Attributes:
        start: When it actually happens.
        original_start: The slot it occupies in its series — its own start for
            an ordinary occurrence, and the replaced slot for a moved one. This
            is the stable identity: keying on `start` instead would orphan the
            stored event every time a showing moved again.
        event: The VEVENT supplying this instance's fields. For a moved
            occurrence that is the override, not the series master.
    """

    start: datetime
    original_start: datetime
    event: VEvent

    @property
    def from_series(self) -> bool:
        """Whether this instance is one of many sharing a UID.

        A caller deriving an identity needs the slot only when the UID alone
        cannot tell tonight's occurrence from next week's.
        """
        return bool(
            self.event.repeated.get("RRULE") or self.event.repeated.get("RECURRENCE-ID")
        )


def expand_calendar(
    events: list[VEvent],
    window_start: datetime,
    window_end: datetime,
    zone: tzinfo,
    logger: Any = None,
) -> list[Occurrence]:
    """Expand a whole calendar, resolving modified instances against their series.

    ICS expresses a changed instance as a *separate* VEVENT sharing the series
    UID and naming the slot it replaces in `RECURRENCE-ID`. Expanding the two
    independently double-books the showing at both its old and new time, so they
    have to be resolved together.

    Args:
        events: Every VEVENT in the document.
        window_start: Inclusive start of the window.
        window_end: Exclusive end.
        zone: Timezone assumed for naive values declaring no TZID.
        logger: Structured logger for degraded rules. Optional.

    Returns:
        Occurrences in chronological order.
    """
    overrides: dict[tuple[str, datetime], VEvent] = {}
    masters: list[VEvent] = []
    series_uids: set[str] = set()

    for event in events:
        slot = _replaced_slot(event, zone)
        if slot is None:
            masters.append(event)
            if event.uid:
                series_uids.add(event.uid)
        elif event.uid:
            overrides[(event.uid, slot)] = event
        else:
            # No UID means no series to join. Still an event, so keep it.
            masters.append(event)

    occurrences: list[Occurrence] = []
    claimed: set[tuple[str, datetime]] = set()

    for event in masters:
        for start in expand(event, window_start, window_end, zone, logger):
            key = (event.uid or "", start)
            replacement = overrides.get(key)
            if replacement is None:
                occurrences.append(Occurrence(start=start, original_start=start, event=event))
                continue
            claimed.add(key)
            moved = _resolve(replacement, window_start, window_end, zone)
            if moved is not None:
                occurrences.append(
                    Occurrence(start=moved, original_start=start, event=replacement)
                )

    # Two kinds of override remain. One replaces a slot that fell outside the
    # window and moves it *in*, so no master occurrence ever claimed it. The
    # other is an orphan whose series is not in this document at all — a feed
    # is free to ship one without the master.
    for (uid, slot), replacement in overrides.items():
        if (uid, slot) in claimed:
            continue
        if uid in series_uids and window_start <= slot < window_end:
            # Its master expanded over this slot and declined it, which only
            # happens when the override was cancelled or moved away.
            continue
        moved = _resolve(replacement, window_start, window_end, zone)
        if moved is not None:
            occurrences.append(Occurrence(start=moved, original_start=slot, event=replacement))

    return sorted(occurrences, key=lambda occurrence: occurrence.start)


def _resolve(override: VEvent, window_start: datetime, window_end: datetime, zone: tzinfo) -> datetime | None:
    """Where a modified instance actually lands, or None if it no longer applies."""
    if _is_cancelled(override) or override.dtstart is None:
        return None

    moved = _localise(override.dtstart, override.dtstart_tzid, zone)

    return moved if window_start <= moved < window_end else None


def _replaced_slot(event: VEvent, zone: tzinfo) -> datetime | None:
    """The occurrence this VEVENT replaces, or None when it replaces nothing."""
    entries = event.repeated.get("RECURRENCE-ID")
    if not entries:
        return None

    parsed, tzid = parse_timestamp(entries[0].value, entries[0].params)
    if parsed is None:
        return None

    return _localise(parsed, tzid, zone)


def _is_cancelled(event: VEvent) -> bool:
    return (event.status or "").upper() == "CANCELLED"


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
