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
from src.models.ranked_event import RankedEvent
from src.presentation.handles import HANDLE_SIGIL, short_handle
from src.utils.nights import night_of

__all__ = [
    "RankedEvent",
    "after_sunset",
    "dated",
    "during_night",
    "matching",
    "night_of",
    "on_date",
    "overlapping",
    "parse_time_window",
    "undated",
]


def matching(pairs: list[RankedEvent], selector: str) -> list[RankedEvent]:
    """Pairs a `--explain` selector names, in the order given.

    Two kinds, told apart by the sigil:

    - **a `#handle`**, or any prefix of one, as git resolves a short SHA;
    - **a title substring**, case-insensitive, which is what anything else is.

    **A rank is deliberately not a selector.** Displayed numbering is
    view-local, so the number beside an event names a different event under
    every filter, and the number a reader types is the one they *counted* — the
    resulting explanation would be valid, of the wrong event, with nothing to
    signal it. The sigil is what keeps the two apart: an unmarked handle that
    happened to be all digits would reintroduce exactly that ambiguity.

    A consequence worth having: a bare number is now a title substring, so
    `--explain 1984` finds the film. Under the old integer-first rule no title
    containing digits was reachable at all.

    Returns:
        Every match. One is the answer, several is a question for the caller to
        put back to the reader, none is a miss — picking one silently is how
        somebody ends up reading the wrong event's explanation.
    """
    if selector.startswith(HANDLE_SIGIL):
        prefix = selector[len(HANDLE_SIGIL) :].strip().casefold()
        return [
            pair
            for pair in pairs
            if short_handle(pair.event.event_id).startswith(prefix)
        ]

    needle = selector.casefold()
    return [
        pair
        for pair in pairs
        if pair.event.title and needle in pair.event.title.casefold()
    ]

#: One ranked event as the CLI reads it: the run's decision, plus what it decided about.

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


def dated(pairs: list[RankedEvent]) -> list[RankedEvent]:
    """Select pairs whose event has a start time."""
    return [pair for pair in pairs if pair.event.start_time is not None]


def undated(pairs: list[RankedEvent]) -> list[RankedEvent]:
    """Select pairs whose event has no start time.

    These are ranked like any other event; only their timing is unknown, so the
    CLI shows them apart from events it can honestly place on a clock.
    """
    return [pair for pair in pairs if pair.event.start_time is None]


def on_date(pairs: list[RankedEvent], day: date) -> list[RankedEvent]:
    """Select pairs whose event starts on the given local date."""
    return [
        pair
        for pair in pairs
        if pair.event.start_time is not None and pair.event.start_time.date() == day
    ]


def during_night(
    pairs: list[RankedEvent],
    night: date,
    day_starts_at: time,
    zone: tzinfo,
) -> list[RankedEvent]:
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
        start = pair.event.start_time
        if start is None:
            continue
        if start.tzinfo is None:
            # Normalization guarantees aware datetimes, so this is defensive.
            # Comparing naive to aware raises, and one bad row must not take
            # down every other event in the view.
            start = start.replace(tzinfo=zone)
        # No end time means instantaneous rather than an invented duration,
        # matching `overlapping`.
        end = pair.event.end_time or start
        if end.tzinfo is None:
            end = end.replace(tzinfo=zone)
        if start < window_to and end >= window_from:
            kept.append(pair)
    return kept


#: Fallback for `long_span_hours` when no caller supplies one — the view root's
#: config load is deliberately tolerant, so an unreadable config must still
#: produce a usable listing. `config.view.long_span_hours` is the real setting.
DEFAULT_LONG_SPAN_HOURS = 24


def overlapping(
    pairs: list[RankedEvent],
    window_start: time,
    window_end: time,
    *,
    night: date,
    long_span_hours: int = DEFAULT_LONG_SPAN_HOURS,
) -> list[RankedEvent]:
    """Select pairs whose event is on during a time-of-day window.

    The window is anchored to the **night being shown**, not to each event's own
    date. Anchoring it to the event meant that for anything which began earlier
    the window was built on that earlier date and the comparison stopped meaning
    anything — a month-long exhibition that opened on the 1st cleared every
    window unconditionally.

    The window's end is inclusive and the **event's** end is exclusive: you can
    arrive as the window closes, but not at something that has just finished.

    An event with no `end_time` is treated as instantaneous rather than given an
    invented duration. Changing that is a separate decision, because it also
    changes single-day events and `during_night` uses the same idiom — the two
    filters have to agree about one event.
    """
    kept = []
    for pair in pairs:
        start = pair.event.start_time
        if start is None:
            continue
        end = pair.event.end_time or start
        window_from = _combine_on(night, start, window_start)
        window_to = _combine_on(night, start, window_end)

        if _is_long_span(start, end, long_span_hours):
            if _runs_all_day(start, end):
                kept.append(pair)
                continue
            # Its own endpoints are its daily hours. Compared on the night being
            # shown, so a 09:00–12:00 workshop stops matching an evening while a
            # festival running 20:00–23:00 nightly still does.
            start = _combine_on(night, start, start.timetz())
            end = _combine_on(night, start, end.timetz())

        if pair.event.end_time is None:
            # Instantaneous, so a half-open interval would be empty and the
            # exclusive end below would drop an event starting exactly as the
            # window opens. It overlaps if it starts inside the window.
            if window_from <= start <= window_to:
                kept.append(pair)
        elif start <= window_to and end > window_from:
            kept.append(pair)
    return kept


def _is_long_span(start: datetime, end: datetime, long_span_hours: int) -> bool:
    """Whether a span is too long to be one continuous occurrence."""
    return (end - start).total_seconds() > long_span_hours * 3600


def _runs_all_day(start: datetime, end: datetime) -> bool:
    """Whether a long span is genuinely continuous rather than a daily programme.

    Equal times-of-day mean the span is a whole number of days, which is the one
    readable signal of continuity in data that records no recurrence. A mooring
    rental stored 12:00 → 12:00 next day is not a programme that runs at noon —
    you have the mooring at 8pm.

    It will occasionally be wrong: a festival that genuinely runs overnight for
    a week reads as a daily programme. Accepted, because the alternative needs a
    field no source publishes and the behaviour it replaces was "matches every
    window that can be typed".
    """
    return start.timetz() == end.timetz()


def after_sunset(pairs: list[RankedEvent]) -> list[RankedEvent]:
    """Select pairs whose event starts after sunset on its own date.

    Sunset is read from the event's own `astronomical_data` rather than from
    tonight's, so the filter stays correct for events on other dates. An event
    without that data is dropped: we cannot assert it qualifies.
    """
    kept = []
    for pair in pairs:
        start = pair.event.start_time
        sunset = _sunset_of(pair.event)
        if start is not None and sunset is not None and start > sunset:
            kept.append(pair)
    return kept


def _combine(reference: datetime, at: time) -> datetime:
    """Place a time of day on the reference datetime's own date and timezone."""
    return datetime.combine(reference.date(), at, tzinfo=reference.tzinfo)


def _combine_on(night: date, reference: datetime, at: time) -> datetime:
    """Place a time of day on a given date, in the reference's own timezone.

    The zone comes from the event rather than from the caller: a window is a
    claim about local wall-clock hours, and the event already knows which offset
    its own listing was published under.
    """
    return datetime.combine(night, at.replace(tzinfo=None), tzinfo=reference.tzinfo)


def _sunset_of(event: Event) -> datetime | None:
    """Read an event's sunset, or None if it was never enriched with one."""
    raw = (event.astronomical_data or {}).get("sunset")
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw)
    except (TypeError, ValueError):
        return None
