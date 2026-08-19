"""SQLite storage for the per-source identity evidence and latch."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from src.ingestion.id_churn import ChurnTally
from src.storage.identity_state import CHURN_THRESHOLD, IdentityState
from src.storage.sqlite.connection import connect


class SqliteIdentityStateRepository:
    """Reads and writes the per-source identity evidence."""

    def __init__(self, db_path: Path | str) -> None:
        self._db_path = db_path

    def get(self, source: str) -> IdentityState:
        """The state for one source, empty if it has never been measured."""
        with connect(self._db_path) as conn:
            row = conn.execute(
                "SELECT churn_evidence, qualifying_runs, latched_at, updated_at "
                "FROM source_identity_state WHERE source = ?",
                (source,),
            ).fetchone()

        if row is None:
            return IdentityState(source, 0, 0, None, None)

        return IdentityState(
            source=source,
            churn_evidence=row[0],
            qualifying_runs=row[1],
            latched_at=_parse(row[2]),
            updated_at=_parse(row[3]),
        )

    def record(self, source: str, tally: ChurnTally, *, at: datetime) -> None:
        """Add one run's evidence, if that run said anything at all.

        A run qualifies when its rate is measurable and at or above the
        threshold. There is deliberately **no per-run sample-size gate**: a
        minimum of 20 seen listings per run does not make a small feed slower to
        latch, it makes it impossible — three live feeds never see more than
        five listings in a night and would churn at 100% for ever unnoticed. The
        sample requirement belongs on the accumulated total instead.
        """
        rate = tally.rate
        if rate is None or rate < CHURN_THRESHOLD:
            return

        with connect(self._db_path) as conn:
            conn.execute(
                "INSERT INTO source_identity_state "
                "(source, churn_evidence, qualifying_runs, updated_at) "
                "VALUES (?, ?, 1, ?) "
                "ON CONFLICT(source) DO UPDATE SET "
                "  churn_evidence = churn_evidence + excluded.churn_evidence, "
                "  qualifying_runs = qualifying_runs + 1, "
                "  updated_at = excluded.updated_at",
                (source, tally.churned, at.isoformat()),
            )
            conn.commit()

    def latch(self, source: str, *, at: datetime) -> None:
        """Record that this source is permanently on content-derived ids.

        Idempotent, and it keeps the *first* time: the latch is one-way, so a
        later call must not restate when the decision was made.
        """
        with connect(self._db_path) as conn:
            conn.execute(
                "INSERT INTO source_identity_state "
                "(source, latched_at, updated_at) VALUES (?, ?, ?) "
                "ON CONFLICT(source) DO UPDATE SET "
                "  latched_at = COALESCE(latched_at, excluded.latched_at), "
                "  updated_at = excluded.updated_at",
                (source, at.isoformat(), at.isoformat()),
            )
            conn.commit()

    def latched(self) -> set[str]:
        """Every source whose publisher ids have been abandoned."""
        with connect(self._db_path) as conn:
            return {
                row[0]
                for row in conn.execute(
                    "SELECT source FROM source_identity_state "
                    "WHERE latched_at IS NOT NULL"
                )
            }


def _parse(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value) if value else None
