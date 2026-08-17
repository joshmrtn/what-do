"""In-memory `RunRepository` — the single official fake.

Exists so stages and the scheduler can be tested without SQLite while still
being held to the same contract. See `tests/unit/storage/test_run_repository.py`.
"""

from __future__ import annotations

import uuid
from dataclasses import replace
from datetime import datetime

from src.models.run import RunRecord


class InMemoryRunRepository:
    """Holds run records in a dict, keyed by run id."""

    def __init__(self) -> None:
        self._runs: dict[str, RunRecord] = {}

    def start(
        self,
        started_at: datetime,
        scoring_config: str | None = None,
        dedup_config: str | None = None,
        preference_revision_id: str | None = None,
    ) -> str:
        """Record that a batch has begun, returning its run id.

        `dedup_config` is accepted and not stored, matching `RunRecord`, which
        has no field for it either. Taking the argument is what matters: without
        it this fake refused a call the protocol allows and the SQLite
        repository accepts.
        """
        run_id = str(uuid.uuid4())
        self._runs[run_id] = RunRecord(
            run_id=run_id,
            started_at=started_at,
            scoring_config=scoring_config,
            preference_revision_id=preference_revision_id,
        )
        return run_id

    def finish(
        self,
        run_id: str,
        *,
        outcome: str,
        completed_at: datetime,
        stage_counts: dict[str, int] | None = None,
        errors: list[str] | None = None,
        skipped_sources: list[str] | None = None,
    ) -> None:
        """Complete a run's record. An unknown id updates nothing."""
        existing = self._runs.get(run_id)
        if existing is None:
            return

        elapsed = completed_at - existing.started_at
        self._runs[run_id] = replace(
            existing,
            completed_at=completed_at,
            duration_ms=int(elapsed.total_seconds() * 1000),
            outcome=outcome,
            stage_counts=dict(stage_counts or {}),
            errors=list(errors or []),
            skipped_sources=list(skipped_sources or []),
        )

    def open_run(self) -> RunRecord | None:
        """The most recent run that began and never finished, if any."""
        unfinished = [r for r in self._runs.values() if r.completed_at is None]
        if not unfinished:
            return None
        return max(unfinished, key=lambda record: record.started_at)

    def get(self, run_id: str) -> RunRecord | None:
        """One run's record, or None if no such run exists."""
        return self._runs.get(run_id)
