"""A record that a stored ranking was recomputed after the batch that made it."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime


@dataclass(frozen=True)
class Rescore:
    """One read-time recomputation of one run's ordering.

    A row per rescore rather than a column that the next one overwrites. After a
    rescore, a run date holds numbers produced by a different forecast from the
    one its own `run_history` row describes, and this is the only thing that
    says so — a `rescored_at` column would answer "when last?" while destroying
    the answer to "how did this move, and how often?".

    Attributes:
        run_date: The run whose ordering was replaced. Always the loaded run's
            own date, never today's: the batch names a run by calendar date and
            the view names a night by `night_of`, and the two disagree either
            side of midnight.
        rescored_at: When the recomputation ran.
        forecast_issued_at: When the forecast it scored against was issued, or
            None where no event carried one — an all-indoor listing rather than
            a failure.
        preference_revision_id: Which preference revision it scored against.
        events_rescored: How many events were re-ranked.
    """

    run_date: date
    rescored_at: datetime
    events_rescored: int
    forecast_issued_at: datetime | None = None
    preference_revision_id: str | None = None
