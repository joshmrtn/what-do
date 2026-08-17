"""In-memory `RescoreRepository` — the single official fake."""

from __future__ import annotations

from datetime import date

from src.models.rescore import Rescore


class InMemoryRescoreRepository:
    """Holds rescores in a list, appended to and never replaced."""

    def __init__(self) -> None:
        self._rescores: list[Rescore] = []

    def record(self, rescore: Rescore) -> None:
        """Append one rescore. Never an update — the previous rows stay."""
        self._rescores.append(rescore)

    def latest_for(self, run_date: date) -> Rescore | None:
        """The most recent rescore of one run, or None if it has never been."""
        found = self.for_run(run_date)
        return found[0] if found else None

    def for_run(self, run_date: date) -> list[Rescore]:
        """Every rescore of one run, newest first.

        Sorted on the stamp rather than returned in write order, so a rescore
        recorded out of sequence does not claim to be the newest — the SQLite
        repository orders the same way.
        """
        return sorted(
            (r for r in self._rescores if r.run_date == run_date),
            key=lambda r: r.rescored_at,
            reverse=True,
        )
