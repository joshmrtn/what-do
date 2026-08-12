"""SQLite-backed `EntityRepository`."""

from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from src.models.candidate_entity import ACTIVE, PROBATIONARY, CandidateEntity
from src.storage.db import connect, has_schema

_COLUMNS = (
    "id, handle, state, depth, mention_count, mention_sources, "
    "llm_classification, discovery_context, promoted_venue_id, "
    "created_at, updated_at"
)


def _to_entity(row: tuple[Any, ...]) -> CandidateEntity:
    """Map one `candidate_entities` row onto a CandidateEntity."""
    return CandidateEntity(
        entity_id=row[0],
        handle=row[1],
        state=row[2],
        depth=row[3],
        mention_count=row[4],
        mention_sources=json.loads(row[5]) if row[5] else [],
        llm_classification=row[6],
        discovery_context=row[7],
        promoted_venue_id=row[8],
        created_at=datetime.fromisoformat(row[9]) if row[9] else None,
        updated_at=datetime.fromisoformat(row[10]) if row[10] else None,
    )


class SqliteEntityRepository:
    """Reads and writes `candidate_entities`."""

    def __init__(self, db_path: Path | str) -> None:
        self._db_path = db_path

    def active_handles(self) -> list[str]:
        """Every handle currently active for ingestion, alphabetically."""
        if not has_schema(self._db_path):
            return []

        conn = connect(self._db_path)
        try:
            rows = conn.execute(
                f"SELECT handle FROM candidate_entities WHERE state = '{ACTIVE}' "
                "ORDER BY handle"
            ).fetchall()
        finally:
            conn.close()

        return [row[0] for row in rows]

    def mark_seeds_active(self, handles: list[str], *, now: datetime) -> None:
        """Upsert seed handles as active at depth 0, keeping any counters."""
        if not handles:
            return

        stamp = now.isoformat()
        conn = connect(self._db_path)
        try:
            for handle in handles:
                existing = conn.execute(
                    "SELECT id FROM candidate_entities WHERE handle = ?", (handle,)
                ).fetchone()
                if existing:
                    conn.execute(
                        "UPDATE candidate_entities "
                        "SET state = ?, depth = 0, updated_at = ? WHERE handle = ?",
                        (ACTIVE, stamp, handle),
                    )
                else:
                    conn.execute(
                        "INSERT INTO candidate_entities "
                        "(id, handle, state, depth, mention_count, mention_sources, "
                        " created_at, updated_at) "
                        "VALUES (?, ?, ?, 0, 0, '[]', ?, ?)",
                        (str(uuid.uuid4()), handle, ACTIVE, stamp, stamp),
                    )
            conn.commit()
        finally:
            conn.close()

    def record_mention(
        self,
        *,
        handle: str,
        source_handle: str,
        depth: int,
        context: str | None,
        now: datetime,
    ) -> None:
        """Record a mention, accumulating onto any handle already seen."""
        stamp = now.isoformat()
        conn = connect(self._db_path)
        try:
            row = conn.execute(
                "SELECT id, mention_count, mention_sources FROM candidate_entities "
                "WHERE handle = ?",
                (handle,),
            ).fetchone()

            if row is None:
                conn.execute(
                    "INSERT INTO candidate_entities "
                    "(id, handle, state, depth, mention_count, mention_sources, "
                    " discovery_context, created_at, updated_at) "
                    "VALUES (?, ?, ?, ?, 1, ?, ?, ?, ?)",
                    (
                        str(uuid.uuid4()),
                        handle,
                        PROBATIONARY,
                        depth,
                        json.dumps([source_handle]),
                        context,
                        stamp,
                        stamp,
                    ),
                )
            else:
                entity_id, count, sources_json = row
                sources: list[str] = json.loads(sources_json) if sources_json else []
                if source_handle in sources:
                    return
                sources.append(source_handle)
                conn.execute(
                    "UPDATE candidate_entities "
                    "SET mention_count = ?, mention_sources = ?, "
                    "    discovery_context = COALESCE(discovery_context, ?), "
                    "    updated_at = ? "
                    "WHERE id = ?",
                    (count + 1, json.dumps(sources), context, stamp, entity_id),
                )
            conn.commit()
        finally:
            conn.close()

    def by_handle(self, handle: str) -> CandidateEntity | None:
        """One entity by its handle, or None if it has never been seen."""
        found = self._select("WHERE handle = ?", (handle,))
        return found[0] if found else None

    def unclassified(self) -> list[CandidateEntity]:
        """Probationary handles disambiguation has not yet judged."""
        return self._select(
            "WHERE state = ? AND llm_classification IS NULL", (PROBATIONARY,)
        )

    def classify(
        self, entity_id: str, *, classification: str, state: str, now: datetime
    ) -> None:
        """Record what disambiguation decided, and the state that follows."""
        conn = connect(self._db_path)
        try:
            conn.execute(
                "UPDATE candidate_entities "
                "SET llm_classification = ?, state = ?, updated_at = ? WHERE id = ?",
                (classification, state, now.isoformat(), entity_id),
            )
            conn.commit()
        finally:
            conn.close()

    def awaiting_promotion(self) -> list[CandidateEntity]:
        """Probationary handles classified as venues, with their evidence."""
        return self._select(
            "WHERE state = ? AND llm_classification = 'venue'", (PROBATIONARY,)
        )

    def activate(self, entity_id: str, *, now: datetime) -> None:
        """Promote a handle to `active`."""
        conn = connect(self._db_path)
        try:
            conn.execute(
                "UPDATE candidate_entities SET state = ?, updated_at = ? WHERE id = ?",
                (ACTIVE, now.isoformat(), entity_id),
            )
            conn.commit()
        finally:
            conn.close()

    def _select(self, where: str, params: tuple[Any, ...]) -> list[CandidateEntity]:
        conn = connect(self._db_path)
        try:
            rows = conn.execute(
                f"SELECT {_COLUMNS} FROM candidate_entities {where} ORDER BY handle",
                params,
            ).fetchall()
        finally:
            conn.close()
        return [_to_entity(row) for row in rows]
