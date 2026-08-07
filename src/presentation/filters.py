"""View filters for the CLI.

Pure functions over already-ranked pairs: no I/O, no clock of their own, and no
reordering. A filter may drop a pair, never move one — the batch's rank order is
the product, and these run after it.

Every filter here is a claim about *when* an event happens, so an event with no
`start_time` fails all of them. That is not the same as hiding it: undated
events are selected by `undated()` and rendered under their own heading.
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta, tzinfo

from src.models.event import Event
from src.utils.nights import night_of
from src.models.recommendation import Recommendation

__all__ = [
    "RankedPair",
    "after_sunset",
    "dated",
    "during_night",
    "night_of",
    "on_date",
    "overlapping",
    "parse_time_window",
    "undated",
]

#: One ranked event as the CLI reads it: the run's decision, plus what it decided about.
RankedPair = tuple[Recommendation, Event]

_WINDOW_FORMAT = "HH:MM-HH:MM"


def parse_time_window(spec: str) -> tuple[time, time]:
    """Parse a `--time` argument into a start and end time of day.

    Args:
        spec: A window such as "20:30-23:30".

    Returns:
        The window's start and end as times of day.

    Raises:
        ValueError: If the window is malformed or crosses midnight. Wrapping is
            not supported in v1, and silently returning the inverse window would
            answer a different question than the one asked.
    """
    parts = spec.split("-")
    if len(parts) != 2:
        raise ValueError(f"Invalid time window {spec!r}: expected {_WINDOW_FORMAT}")

    try:
        start = time.fromisoformat(parts[0].strip())
        end = time.fromisoformat(parts[1].strip())
    except ValueError as exc:
        raise ValueError(f"Invalid time window {spec!r}: expected {_WINDOW_FORMAT}") from exc

    if start >= end:
        raise ValueError(
            f"Invalid time window {spec!r}: {_WINDOW_FORMAT} must not cross midnight"
        )

    return start, end


def dated(pairs: list[RankedPair]) -> list[RankedPair]:
    """Select pairs whose event has a start time."""
    return [pair for pair in pairs if pair[1].start_time is not None]


def undated(pairs: list[RankedPair]) -> list[RankedPair]:
    """Select pairs whose event has no start time.

    These are ranked like any other event; only their timing is unknown, so the
    CLI shows them apart from events it can honestly place on a clock.
    """
    return [pair for pair in pairs if pair[1].start_time is None]


def on_date(pairs: list[RankedPair], day: date) -> list[RankedPair]:
    """Select pairs whose event starts on the given local date."""
    return [
        pair
        for pair in pairs
        if pair[1].start_time is not None and pair[1].start_time.date() == day
    ]


def during_night(
    pairs: list[RankedPair],
    night: date,
    day_starts_at: time,
    zone: tzinfo,
) -> list[RankedPair]:
    """Select pairs whose event falls in the night named by `night`.

    The window runs from `night` at `day_starts_at` to the same wall-clock time
    the next day, half-open so two consecutive nights can never both claim the
    same event. It is a wall-clock span, not a fixed 24 hours, so a night
    crossing a DST boundary is honestly 23 or 25 hours long.

    This cannot be expressed as a filter on the event's date: a 00:30 show
    carries the *next* calendar date, and dropping it would empty the evening
    still in progress.

    An event is matched on whether it *overlaps* the night, not on whether it
    starts in it. A month-long exhibition is on every night it is open, and
    asking only about its start would show it on opening night and then never
    again. It stays one stored event either way — this decides which nights it
    appears on, never how many times.

    Args:
        pairs: Ranked pairs, in the batch's rank order.
        night: The date the night is named for.
        day_starts_at: Local time of day at which the night begins and ends.
        zone: The view's timezone, which anchors the window.

    Returns:
        The pairs overlapping the window, order preserved, each at most once.
    """
    window_from = datetime.combine(night, day_starts_at, tzinfo=zone)
    window_to = window_from + timedelta(days=1)

    kept = []
    for pair in pairs:
        start = pair[1].start_time
        if start is None:
            continue
        if start.tzinfo is None:
            # Normalization guarantees aware datetimes, so this is defensive.
            # Comparing naive to aware raises, and one bad row must not take
            # down every other event in the view.
            start = start.replace(tzinfo=zone)
        # No end time means instantaneous rather than an invented duration,
        # matching `overlapping`.
        end = pair[1].end_time or start
        if end.tzinfo is None:
            end = end.replace(tzinfo=zone)
        if start < window_to and end >= window_from:
            kept.append(pair)
    return kept


def overlapping(pairs: list[RankedPair], window_start: time, window_end: time) -> list[RankedPair]:
    """Select pairs whose event overlaps a time-of-day window, both ends inclusive.

    The window is anchored to each event's own date, so it means the same thing
    for tonight and for an event three days out. An event with no `end_time` is
    treated as instantaneous rather than being given an invented duration: it
    overlaps only if it starts inside the window.
    """
    kept = []
    for pair in pairs:
        start = pair[1].start_time
        if start is None:
            continue
        window_from = _combine(start, window_start)
        window_to = _combine(start, window_end)
        end = pair[1].end_time or start
        if start <= window_to and end >= window_from:
            kept.append(pair)
    return kept


def after_sunset(pairs: list[RankedPair]) -> list[RankedPair]:
    """Select pairs whose event starts after sunset on its own date.

    Sunset is read from the event's own `astronomical_data` rather than from
    tonight's, so the filter stays correct for events on other dates. An event
    without that data is dropped: we cannot assert it qualifies.
    """
    kept = []
    for pair in pairs:
        start = pair[1].start_time
        sunset = _sunset_of(pair[1])
        if start is not None and sunset is not None and start > sunset:
            kept.append(pair)
    return kept


def _combine(reference: datetime, at: time) -> datetime:
    """Place a time of day on the reference datetime's own date and timezone."""
    return datetime.combine(reference.date(), at, tzinfo=reference.tzinfo)


def _sunset_of(event: Event) -> datetime | None:
    """Read an event's sunset, or None if it was never enriched with one."""
    raw = (event.astronomical_data or {}).get("sunset")
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw)
    except (TypeError, ValueError):
        return None
