"""Accumulated evidence about a source's identifiers, and the one-way latch.

The churn detector has measured every source since it shipped, and reported into
a run summary nobody is obliged to read. This is what lets the measurement act:
evidence accumulates across runs until it is decisive, and then the source is
moved onto content-derived ids permanently.

**The state is derived, and config is not.** What is stored here is a
measurement outcome — recomputable from `event_candidates` history — so it
belongs in the database. The *policy* (`auto`, `publisher`, `content`) is the
human's declaration and stays in config. They are different things, not two
copies of one.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

from src.ingestion.id_churn import ChurnTally


@dataclass(frozen=True)
class IdentityState:
    """What has been observed about one source's identifiers."""

    source: str
    #: Churned listings summed over every qualifying run. Never reset.
    churn_evidence: int
    #: How many runs contributed. Guards against a single anomalous run.
    qualifying_runs: int
    #: When the latch fired, or None. One-way: never cleared.
    latched_at: datetime | None
    updated_at: datetime | None


#: A run counts as evidence at or above this rate. Not a delicate number:
#: measured over the live corpus, sixteen sources reported either 100.0% or
#: exactly 0.0% with nothing in between, and a genuine republish
#: (`cinemasalem`, one listing of seventy) sits at ~1.4%. Every threshold
#: between 0.5 and 0.8 gave an identical answer on every night of real data.
CHURN_THRESHOLD = 0.5


class IdentityStateStore(Protocol):
    """What the latch needs from storage. A protocol so the latch depends on the
    behaviour rather than on SQLite."""

    def get(self, source: str) -> IdentityState:
        """The state for one source, empty if never measured."""
        ...

    def record(self, source: str, tally: ChurnTally, *, at: datetime) -> None:
        """Add one run's evidence, if that run said anything at all."""
        ...

    def latch(self, source: str, *, at: datetime) -> None:
        """Record that this source is permanently on content-derived ids."""
        ...

    def latched(self) -> set[str]:
        """Every source whose publisher ids have been abandoned."""
        ...
