"""Event persistence and reload.

The batch computes expensive artifacts — LLM-extracted tags and summaries,
embeddings — that are worthless if they die with the process. Persisting them
also activates the skip-if-done branches in `ExtractionStage` and
`EmbeddingStage`, which makes a re-run incremental: only new events cost model
time. LLM extraction runs about three minutes per event locally, so that is the
difference between a re-run costing minutes and costing hours.

The CLI depends on this too: it reads precomputed data and never calls a model.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

from src.models.event import Event
from src.models.tag import tags_from_json, tags_to_json
from src.utils.vectors import pack_vectors, unpack_vectors

EVENT_COLUMNS = (
    "id, source_event_candidates, source_type, url, image_url, title, venue, "
    "description, location, start_time, end_time, tags, summary, "
    "tag_embeddings, summary_embedding, weather, astronomical_data, metadata, "
    "created_at, updated_at, setting"
)


def event_to_row(event: Event) -> tuple[Any, ...]:
    """Flatten an Event into a row for the events table.

    `image_bytes` is deliberately not stored: it exists only to feed the
    multimodal extraction call, which a reloaded event skips anyway, and image
    blobs would bloat the database for no downstream reader.
    """
    return (
        event.event_id,
        json.dumps(event.source_event_candidates),
        event.source_type,
        event.url,
        event.image_url,
        event.title,
        event.venue,
        event.description,
        event.location,
        event.start_time.isoformat() if event.start_time else None,
        event.end_time.isoformat() if event.end_time else None,
        tags_to_json(event.tags),
        event.summary,
        pack_vectors(event.tag_embeddings) if event.tag_embeddings else None,
        event.summary_embedding,
        json.dumps(event.weather) if event.weather is not None else None,
        json.dumps(event.astronomical_data) if event.astronomical_data is not None else None,
        json.dumps(event.metadata),
        event.created_at.isoformat(),
        event.updated_at.isoformat(),
        event.setting,
    )


def row_to_event(row: tuple[Any, ...]) -> Event:
    """Rebuild an Event from a row selected with EVENT_COLUMNS."""
    tags = tags_from_json(row[11]) if row[11] else []
    packed = row[13]

    return Event(
        event_id=row[0],
        source_event_candidates=json.loads(row[1]) if row[1] else [],
        source_type=row[2],
        url=row[3],
        image_url=row[4],
        title=row[5],
        venue=row[6],
        description=row[7],
        location=row[8],
        start_time=_parse_dt(row[9]),
        end_time=_parse_dt(row[10]),
        tags=tags,
        summary=row[12],
        tag_embeddings=unpack_vectors(packed, count=len(tags)) if packed and tags else [],
        summary_embedding=row[14],
        weather=json.loads(row[15]) if row[15] else None,
        astronomical_data=json.loads(row[16]) if row[16] else None,
        metadata=json.loads(row[17]) if row[17] else {},
        created_at=_require_dt(row[18]),
        updated_at=_require_dt(row[19]),
        setting=row[20] or "unknown",
    )


def _parse_dt(value: str | None) -> datetime | None:
    """Parse an optional stored ISO timestamp, preserving its offset."""
    return datetime.fromisoformat(value) if value else None


def _require_dt(value: str) -> datetime:
    """Parse a stored ISO timestamp that must be present."""
    return datetime.fromisoformat(value)


def save_events(events: list[Event], db_path: Path | str) -> None:
    """Insert or update events, preserving tags, embeddings, and enrichment.

    Args:
        events: Events to persist. Existing rows are replaced by event_id.
        db_path: Path to the SQLite database.
    """
    if not events:
        return

    placeholders = ", ".join("?" * len(EVENT_COLUMNS.split(", ")))
    conn = sqlite3.connect(db_path)
    try:
        conn.executemany(
            f"INSERT OR REPLACE INTO events ({EVENT_COLUMNS}) VALUES ({placeholders})",
            [event_to_row(e) for e in events],
        )
        conn.commit()
    finally:
        conn.close()


def delete_events(event_ids: list[str], db_path: Path | str) -> None:
    """Remove events superseded by a merge.

    Args:
        event_ids: Events to delete. An empty list is a no-op — nothing was
            superseded, which is never the same as "clear the table".
        db_path: Path to the SQLite database.
    """
    if not event_ids:
        return

    conn = sqlite3.connect(db_path)
    try:
        conn.executemany("DELETE FROM events WHERE id = ?", [(i,) for i in event_ids])
        conn.commit()
    finally:
        conn.close()


def load_events(db_path: Path | str) -> list[Event]:
    """Load all persisted events.

    Args:
        db_path: Path to the SQLite database.

    Returns:
        Events with tags, embeddings, and enrichment restored. `similarity` is
        not stored — it is derived, cheap to recompute, and owned by the
        recommendations table.
    """
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(f"SELECT {EVENT_COLUMNS} FROM events").fetchall()
    finally:
        conn.close()

    return [row_to_event(row) for row in rows]
