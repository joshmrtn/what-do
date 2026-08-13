"""One batch run's durable record."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass(frozen=True)
class RunRecord:
    """What a 2am run did, as stored in `run_history`.

    A record with a `started_at` and no `completed_at` is a run that died
    mid-flight. That state is the reason the row is written at the start rather
    than the end: no end-of-run write can record a process that was killed.

    Attributes:
        run_id: Identifier handed back by `RunRepository.start`.
        started_at: When the batch began.
        completed_at: When it ended, or None if it never did.
        duration_ms: Elapsed time, derived against the stored `started_at` so a
            resumed process still records real elapsed time. None until finished.
        outcome: One of `success`, `partial`, `failed`. None until finished.
        stage_counts: Per-stage counts.
        errors: Stage failure messages.
        skipped_sources: Sources not built, normally for a missing credential.
            Kept apart from `errors` because a skip is a legitimate deployment
            state, and folding the two together loses the distinction the
            credential policy rests on.
        scoring_config: The scoring constants in force, as JSON. None for a
            run that recorded none.
    """

    run_id: str
    started_at: datetime
    completed_at: datetime | None = None
    duration_ms: int | None = None
    outcome: str | None = None
    stage_counts: dict[str, int] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    skipped_sources: list[str] = field(default_factory=list)
    #: The scoring constants this run used, as JSON. `config.yaml` is
    #: gitignored, so without this a past score cannot be explained once a
    #: constant is tuned — and the tag confidence curve is expected to be
    #: re-fitted as nights accumulate.
    scoring_config: str | None = None
