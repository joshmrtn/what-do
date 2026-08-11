"""Reads that span more than one aggregate.

A repository owns one aggregate. The view the CLI renders needs three — the
event, its score and its placement — so composing them belongs here rather than
on any one repository. Putting it on, say, the ranking repository would mean
that repository reading the event store, which is the boundary violation the
repository split exists to remove.

Nothing here holds a connection or writes SQL. It takes repositories and joins
their results in memory.
"""

from __future__ import annotations

from datetime import date

from src.models.ranked_event import RankedEvent
from src.storage.protocols import EventRepository, RankingRepository, ScoreRepository


def load_ranked_events(
    events: EventRepository,
    scores: ScoreRepository,
    rankings: RankingRepository,
    run_date: date | None = None,
) -> list[RankedEvent]:
    """One run's ranked events, in the order the batch assigned.

    A ranking whose event or score has since been purged is skipped rather than
    surfacing as a half-empty row: one missing record must not take down the
    whole view.

    Args:
        events: Where canonical events are read from.
        scores: Where the semantic verdicts are read from.
        rankings: Where the placements are read from.
        run_date: Which batch to read. Defaults to the latest ranked night.

    Returns:
        `RankedEvent`s in rank order. Empty if nothing has been ranked yet.
    """
    target = run_date if run_date is not None else rankings.latest_run_date()
    if target is None:
        return []

    placements = rankings.for_run(target)
    if not placements:
        return []

    by_id = {event.event_id: event for event in events.load_all()}
    verdicts = {score.event_id: score for score in scores.for_run(target)}

    return [
        RankedEvent(
            event=by_id[placement.event_id],
            score=verdicts[placement.event_id],
            ranking=placement,
        )
        for placement in placements
        if placement.event_id in by_id and placement.event_id in verdicts
    ]
