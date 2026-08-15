"""What the refit concluded, as the next run reads it."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass(frozen=True)
class CurveState:
    """The tag-confidence curve in force, and how it got there.

    Written at the end of a batch and read by the *next* one, so no night is
    scored with constants that moved underneath it. Absent means the config
    defaults stand — the state of a fresh deployment, and of any regime that
    has not armed.
    """

    cap: float
    saturation: float
    regime: str | None
    updated_at: datetime
    #: Row counts, both held-out scores, the per-source multipliers and any
    #: change points. A stored score is only interpretable against the fit that
    #: produced it, and "the gate said no" belongs in the record as much as a
    #: move does.
    provenance: dict[str, Any] = field(default_factory=dict)
