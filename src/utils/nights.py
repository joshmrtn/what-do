"""The night boundary, shared by ingestion and presentation.

A "night" is the system's unit of a day. It runs from `day_starts_at` to the
same wall clock the next day, so someone asking at 00:30 means the evening they
are standing in rather than the one beginning in three hours.

This lives outside both packages because they have to agree. When ingestion
floors its window at the current instant and the CLI shows a night, a re-run at
20:00 discards the events the CLI is still displaying — and the events would be
gone from storage, not merely absent from one view.
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta, tzinfo


def night_of(moment: datetime, day_starts_at: time) -> date:
    """Name the night a moment falls in.

    Args:
        moment: The time in question, **already expressed in the relevant
            zone**. Converting is the caller's job, since only it knows which.
        day_starts_at: Local time of day at which one night gives way to the next.

    Returns:
        The date the night is named for.
    """
    if moment.time() < day_starts_at:
        return moment.date() - timedelta(days=1)
    return moment.date()


def night_start(now: datetime, day_starts_at: time, zone: tzinfo) -> datetime:
    """The moment the current night began, in the given zone.

    The floor for anything that keeps events by their own start time. It never
    lands in the future, so an event already under way is never discarded for
    having started before the job that is looking at it.

    Args:
        now: The current time, in any zone.
        day_starts_at: Local time of day at which one night gives way to the next.
        zone: The zone whose wall clock defines the night.

    Returns:
        An aware datetime at `day_starts_at` on the current night's date.
    """
    local = now.astimezone(zone)

    return datetime.combine(night_of(local, day_starts_at), day_starts_at, tzinfo=zone)
