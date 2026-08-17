"""How stale the thing on screen is, and whether anything can be done about it.

Separate from `staleness_notice`, which answers a third question — *is this
ranking from an older night than the one being shown?* That one is about a batch
that did not run. These two are about a batch that ran and whose inputs have
since moved: the forecast has aged, or the preference files have been edited.

Both are recoverable in seconds, which is why they are worth saying out loud
rather than silently tolerating.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Iterable

from src.models.event import Event
from src.models.preference_revision import PreferenceRevision

#: The preference files match the revision the last ranking was scored against.
PREFERENCES_UNCHANGED = "unchanged"
#: They do not, so the order on screen answers a question no longer being asked.
PREFERENCES_CHANGED = "changed"
#: Nothing was recorded, so the question cannot be answered. Deliberately its
#: own state rather than folding into "unchanged": an absent record and a
#: matching one reading the same is the defect that left every weather
#: adjustment at 0.0 for twelve days.
PREFERENCES_UNKNOWN = "unknown"

#: Below this, an age reads better in minutes. "1 hours old" is the kind of
#: wrong that makes a reader distrust the rest of the line.
_MINUTES_BELOW = timedelta(hours=2)


def latest_forecast(events: Iterable[Event]) -> datetime | None:
    """When the freshest forecast behind these events was issued.

    The newest rather than the oldest: the listing is as current as the most
    recent forecast it was scored against, and an event beyond the forecast
    horizon carries an old one forever without making tonight's stale.

    Returns:
        The newest `issued_at`, or None when no event carries a forecast —
        which is an all-indoor listing, not a stale one.
    """
    issued: list[datetime] = []
    for event in events:
        forecast = (event.weather or {}).get("forecast") or {}
        stamp = forecast.get("issued_at")
        if stamp:
            issued.append(datetime.fromisoformat(stamp))
    return max(issued) if issued else None


def preference_state(
    current_hash: str, recorded: PreferenceRevision | None
) -> str:
    """Whether the preference files still say what the last ranking scored on.

    Args:
        current_hash: Content hash of the files as they are now.
        recorded: The newest revision any run recorded, or None if none has.

    Returns:
        One of `PREFERENCES_UNCHANGED`, `PREFERENCES_CHANGED`,
        `PREFERENCES_UNKNOWN`.
    """
    if recorded is None:
        return PREFERENCES_UNKNOWN
    return (
        PREFERENCES_UNCHANGED
        if recorded.content_hash == current_hash
        else PREFERENCES_CHANGED
    )


def _describe_age(age: timedelta) -> str:
    """An age in the unit that reads honestly at that scale."""
    if age < _MINUTES_BELOW:
        return f"{int(age.total_seconds() // 60)} minutes"
    return f"{int(age.total_seconds() // 3600)} hours"


def freshness_notice(
    *,
    forecast_issued_at: datetime | None,
    now: datetime,
    ttl: timedelta,
    preferences: str,
) -> str | None:
    """What is out of date about the listing, or None when nothing is.

    Args:
        forecast_issued_at: When the ranking's freshest forecast was issued.
        now: The current instant.
        ttl: How old a forecast may be before it is worth refreshing.
        preferences: One of the `PREFERENCES_*` states.

    Returns:
        A notice naming each stale input, or None. `PREFERENCES_UNKNOWN` is
        never reported: saying "changed" about a run that recorded nothing
        would be a guess, and saying "unchanged" would be the same guess with
        more confidence.
    """
    lines: list[str] = []

    if forecast_issued_at is not None:
        age = now - forecast_issued_at
        # A forecast stamped in the future is clock skew, not staleness.
        if age > ttl:
            lines.append(f"the forecast behind it is {_describe_age(age)} old")

    if preferences == PREFERENCES_CHANGED:
        lines.append("your preferences have changed since it was scored")

    if not lines:
        return None

    return "⚠  This ranking is out of date: " + ", and ".join(lines) + "."
