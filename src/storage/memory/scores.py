"""In-memory `ScoreRepository` — the single official fake."""

from __future__ import annotations

from datetime import date

from src.models.event_score import EventScore


class InMemoryScoreRepository:
    """Holds scores in a dict keyed by run date."""

    def __init__(self) -> None:
        self._by_run: dict[date, list[EventScore]] = {}

    def save(self, scores: list[EventScore]) -> None:
        """Insert scores for one or more run dates, replacing those runs."""
        if not scores:
            return

        for run_date in {s.run_date for s in scores}:
            self._by_run[run_date] = []
        for score in scores:
            self._by_run[score.run_date].append(score)

    def for_run(self, run_date: date) -> list[EventScore]:
        """Every score stored for one batch date."""
        return list(self._by_run.get(run_date, []))
