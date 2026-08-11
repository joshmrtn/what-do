"""An event with the score and the placement that go with it."""

from __future__ import annotations

from dataclasses import dataclass

from src.models.event import Event
from src.models.event_score import EventScore
from src.models.ranking import Ranking


@dataclass(frozen=True)
class RankedEvent:
    """One row of the view: the event, why it scored, and where it landed.

    A read model rather than something the pipeline produces. It composes three
    aggregates that are deliberately stored apart, which is why nothing writes
    one — the query layer assembles it and the CLI renders it.

    Attributes:
        event: The event itself.
        score: The semantic verdict, carrying the reasons behind it.
        ranking: Its place in the night's order.
    """

    event: Event
    score: EventScore
    ranking: Ranking

    @property
    def rank(self) -> int:
        """Shorthand for the placement, which every caller sorts and prints."""
        return self.ranking.rank
