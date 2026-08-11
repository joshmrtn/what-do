"""What the scorer decided about one event, under one run's preferences."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

from src.scoring.similarity import Reason


@dataclass(frozen=True)
class EventScore:
    """The semantic verdict on an event, independent of tonight's ordering.

    Exists for *every* scored event, whether or not it was in scope to be
    ranked. That is the point of the split: scoring an event and placing it in
    a night's order are different questions, and the second one only applies to
    events happening in the window.

    Keyed by `(event_id, run_date)` rather than by event alone, because
    preferences change — the same event scores differently on different nights,
    and a stored score is only interpretable against the revision that produced
    it.

    Attributes:
        event_id: The event this scores.
        run_date: The batch date that produced it.
        tag_score: Balanced mean over the weighted tag contributions.
        summary_score: The same formula over the one-sentence summary.
        base_score: `tag_score + summary_weight * summary_score`.
        tag_confidence: How much of the expected tag count the extraction
            produced. Belongs here rather than with the ranking: it is a pure
            function of the event's own tags and says nothing about tonight.
        match: "yes", "maybe", or "no" — the advisory classification.
        reasons: Every contribution behind the score.
    """

    event_id: str
    run_date: date
    base_score: float
    match: str
    tag_score: float = 0.0
    summary_score: float = 0.0
    tag_confidence: float = 1.0
    reasons: list[Reason] = field(default_factory=list)
