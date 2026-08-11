"""One event's place in one night's order."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True)
class Ranking:
    """Where an event landed in a run's ordering.

    Frozen because it records what a run decided. `rank` is stored rather than
    recomputed at read time, so a reader cannot accidentally reorder the batch's
    decision, and a later "why did this move?" is answerable from history.

    Only in-scope events get one. An event scored but outside the night's window
    has an `EventScore` and no `Ranking` — which is exactly the distinction that
    lets scores be kept for events the run never ranked.

    Attributes:
        event_id: The event this ranks.
        run_date: The batch date that produced it.
        weather_adjustment: Signed comfort adjustment; 0.0 unless the event is
            outdoors. Belongs here rather than with the score: it depends on
            *tonight's* forecast, not on the event alone.
        final_score: What the ordering is actually built on.
        rank: 1-based position within this run.
    """

    event_id: str
    run_date: date
    final_score: float
    rank: int
    weather_adjustment: float = 0.0
