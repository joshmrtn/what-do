"""HandleExtractor — extracts @handles from post text and updates candidate_entities."""

from __future__ import annotations

import json
import re
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_HANDLE_RE = re.compile(r"@[\w.]+")


class HandleExtractor:
    """Parses @handles from post captions and upserts them into candidate_entities."""

    def __init__(
        self,
        db_path: Path,
        max_depth: int,
        blocklist: list[str],
        logger: Any,
    ) -> None:
        self._db_path = db_path
        self._max_depth = max_depth
        self._blocklist = {h.lower() for h in blocklist if h.startswith("@")}
        self._logger = logger

    def process(self, text: str, source_handle: str, source_depth: int) -> None:
        """Extract handles from text and persist new discoveries.

        Args:
            text: Post caption or description to scan.
            source_handle: The handle that published this text (for mention_sources tracking).
            source_depth: Discovery depth of the source handle.
        """
        candidate_depth = source_depth + 1
        if candidate_depth > self._max_depth:
            return

        handles = _HANDLE_RE.findall(text)
        if not handles:
            return

        conn = sqlite3.connect(self._db_path)
        try:
            now = datetime.now(timezone.utc).isoformat()
            for handle in handles:
                handle_lower = handle.lower()
                if handle_lower in self._blocklist:
                    self._logger.info(
                        f"Skipping blocklisted handle: {handle}",
                        component="handle_extractor",
                        duration_ms=0,
                    )
                    continue
                self._upsert(conn, handle, source_handle, candidate_depth, now)
            conn.commit()
        finally:
            conn.close()

    def _upsert(
        self,
        conn: sqlite3.Connection,
        handle: str,
        source_handle: str,
        depth: int,
        now: str,
    ) -> None:
        row = conn.execute(
            "SELECT id, mention_count, mention_sources FROM candidate_entities WHERE handle = ?",
            (handle,),
        ).fetchone()

        if row is None:
            conn.execute(
                """INSERT INTO candidate_entities
                   (id, handle, state, depth, mention_count, mention_sources, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    str(uuid.uuid4()),
                    handle,
                    "probationary",
                    depth,
                    1,
                    json.dumps([source_handle]),
                    now,
                    now,
                ),
            )
        else:
            entity_id, count, sources_json = row
            sources: list[str] = json.loads(sources_json) if sources_json else []
            if source_handle in sources:
                return  # already counted this source
            sources.append(source_handle)
            conn.execute(
                """UPDATE candidate_entities
                   SET mention_count = ?, mention_sources = ?, updated_at = ?
                   WHERE id = ?""",
                (count + 1, json.dumps(sources), now, entity_id),
            )
