"""In-memory `RankingRepository` — the single official fake."""

from __future__ import annotations

from datetime import date

from src.models.ranking import Ranking


class InMemoryRankingRepository:
    """Holds placements in a dict keyed by run date."""

    def __init__(self) -> None:
        self._by_run: dict[date, list[Ranking]] = {}

    def save(self, rankings: list[Ranking]) -> None:
        """Insert placements for one or more run dates, replacing those runs."""
        if not rankings:
            return

        for run_date in {r.run_date for r in rankings}:
            self._by_run[run_date] = []
        for ranking in rankings:
            self._by_run[ranking.run_date].append(ranking)

    def for_run(self, run_date: date) -> list[Ranking]:
        """One batch date's placements, in the rank the batch assigned."""
        return sorted(self._by_run.get(run_date, []), key=lambda r: r.rank)

    def latest_run_date(self) -> date | None:
        """The most recent batch date that produced a ranking, if any."""
        ranked = [run for run, rows in self._by_run.items() if rows]
        return max(ranked) if ranked else None
