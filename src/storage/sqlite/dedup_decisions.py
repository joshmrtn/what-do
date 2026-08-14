"""SQLite-backed storage for dedup decisions."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from src.normalization.decision_sampling import SampledDecision
from src.storage.dedup_decisions import StoredDecision
from src.storage.sqlite.connection import connect

_COLUMNS = (
    "pass_name, record_kind, record_a, record_b, score, verdict, "
    "stratum, sample_denominator, content_hash_a, content_hash_b, "
    "run_id, updated_at"
)


def _to_row(
    sampled: SampledDecision, run_id: str, now: datetime
) -> tuple[str, str, str, str, float, str, str, int, str, str, str, str]:
    """One decision as a row, in pair order.

    The primary key rests on `record_a < record_b`, so a decision arriving the
    other way round would become a *second* row for a pair the table promises
    to hold once. Normalised here rather than trusted from the caller, because
    this is where the key lives — and the fingerprints are swapped with the ids
    they belong to, or the row would describe each side with the other's text.
    """
    decision = sampled.decision
    left = (decision.record_a, decision.content_hash_a)
    right = (decision.record_b, decision.content_hash_b)
    (record_a, hash_a), (record_b, hash_b) = sorted((left, right))

    return (
        decision.pass_name,
        decision.record_kind,
        record_a,
        record_b,
        decision.score,
        decision.verdict,
        sampled.stratum,
        sampled.sample_denominator,
        hash_a,
        hash_b,
        run_id,
        now.isoformat(),
    )


class SqliteDedupDecisionRepository:
    """Reads and writes `dedup_decisions`."""

    def __init__(self, db_path: Path | str) -> None:
        self._db_path = db_path

    def save(
        self, decisions: list[SampledDecision], *, run_id: str, now: datetime
    ) -> None:
        """Store decisions, replacing any previous verdict on the same pair.

        One row per pair per pass, so volume grows with distinct comparisons
        rather than with nights — the same pair is compared again every run it
        survives, and storing each of those would multiply the table by the
        length of an event's life for no added signal.
        """
        if not decisions:
            return

        conn = connect(self._db_path)
        try:
            conn.executemany(
                f"INSERT OR REPLACE INTO dedup_decisions ({_COLUMNS}) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [_to_row(s, run_id, now) for s in decisions],
            )
            conn.commit()
        finally:
            conn.close()

    def load_all(self) -> list[StoredDecision]:
        """Every stored decision, for inspection and for training."""
        conn = connect(self._db_path)
        try:
            rows = conn.execute(
                f"SELECT {_COLUMNS} FROM dedup_decisions "
                "ORDER BY pass_name, record_a, record_b"
            ).fetchall()
        finally:
            conn.close()

        return [StoredDecision(*row) for row in rows]
