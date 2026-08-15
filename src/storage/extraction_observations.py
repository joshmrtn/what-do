"""One extraction, as it happened."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class ExtractionObservation:
    """What the model was asked and what it answered, at one moment.

    `events` holds only the latest, because re-extraction overwrites — so the
    two observations most worth comparing, either side of a prompt change, are
    exactly the ones that never survive there.
    """

    event_id: str
    observed_at: datetime
    chars: int
    tags: int
    model: str | None
    prompt_version: str | None
    degradation: str | None
    source: str | None
    #: Reconstructed from an event rather than recorded as it happened. A later
    #: reader can then tell evidence from inference.
    backfilled: bool = False
