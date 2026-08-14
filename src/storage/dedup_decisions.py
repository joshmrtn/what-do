"""Shared types for dedup decision storage.

The read model lives here rather than in the SQLite implementation so the
protocol can name it without depending on a backend — the same split
`src/storage/events.py` already makes for its row mappers.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StoredDecision:
    """A decision as it comes back out — what a person reads months later."""

    pass_name: str
    record_kind: str
    record_a: str
    record_b: str
    score: float
    verdict: str
    stratum: str
    sample_denominator: int
    content_hash_a: str
    content_hash_b: str
    run_id: str
    updated_at: str
