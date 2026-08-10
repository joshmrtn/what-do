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

from src.storage.db import connect
from datetime import datetime
from pathlib import Path
from typing import Any

from src.config import DEFAULT_EMBEDDING_MODEL
from src.models.event import Event
from src.models.tag import Tag

EVENT_COLUMNS = (
    "id, source_type, url, image_url, title, venue, venue_id, "
    "description, location, start_time, end_time, summary, "
    "summary_embedding, weather, weather_cache_id, astronomical_data, metadata, "
    "created_at, updated_at, setting, timing, extraction_input_hash, "
    "embedding_input_hash"
)


def validate_tag_vectors(event: Event) -> None:
    """Reject an event whose tag vectors cannot be paired with its tags.

    An event with no vectors at all is the ordinary state between extraction and
    embedding. Some vectors but not one per tag is different: the pairing is
    positional, so a short list silently drops the tail and reports success.

    Args:
        event: The event about to be persisted.

    Raises:
        ValueError: If the event has vectors but not one for every tag.
    """
    if event.tag_embeddings and len(event.tag_embeddings) != len(event.tags):
        raise ValueError(
            f"event {event.event_id} has {len(event.tags)} tags but "
            f"{len(event.tag_embeddings)} tag vectors"
        )


def event_to_row(event: Event) -> tuple[Any, ...]:
    """Flatten an Event into a row for the events table.

    Tags, their vectors and the source candidate ids live in their own tables
    and are written alongside by `save_events`, not encoded here.

    `image_bytes` is deliberately not stored: it exists only to feed the
    multimodal extraction call, which a reloaded event skips anyway, and image
    blobs would bloat the database for no downstream reader.
    """
    return (
        event.event_id,
        event.source_type,
        event.url,
        event.image_url,
        event.title,
        event.venue,
        event.venue_id,
        event.description,
        event.location,
        event.start_time.isoformat() if event.start_time else None,
        event.end_time.isoformat() if event.end_time else None,
        event.summary,
        event.summary_embedding,
        json.dumps(event.weather) if event.weather is not None else None,
        event.weather_cache_id,
        json.dumps(event.astronomical_data) if event.astronomical_data is not None else None,
        json.dumps(event.metadata),
        event.created_at.isoformat(),
        event.updated_at.isoformat(),
        event.setting,
        event.timing,
        event.extraction_input_hash,
        event.embedding_input_hash,
    )


def row_to_event(
    row: tuple[Any, ...],
    tags: list[Tag] | None = None,
    tag_embeddings: list[bytes] | None = None,
    source_candidates: list[str] | None = None,
) -> Event:
    """Rebuild an Event from a row selected with EVENT_COLUMNS.

    Args:
        row: The events-table row.
        tags: Tags from `event_tags`, in position order.
        tag_embeddings: Vectors aligned to `tags`, empty when unembedded.
        source_candidates: Candidate ids from `event_source_candidates`.
    """
    return Event(
        event_id=row[0],
        source_event_candidates=source_candidates or [],
        source_type=row[1],
        url=row[2],
        image_url=row[3],
        title=row[4],
        venue=row[5],
        venue_id=row[6],
        description=row[7],
        location=row[8],
        start_time=_parse_dt(row[9]),
        end_time=_parse_dt(row[10]),
        tags=tags or [],
        summary=row[11],
        tag_embeddings=tag_embeddings or [],
        summary_embedding=row[12],
        weather=json.loads(row[13]) if row[13] else None,
        weather_cache_id=row[14],
        astronomical_data=json.loads(row[15]) if row[15] else None,
        metadata=json.loads(row[16]) if row[16] else {},
        created_at=_require_dt(row[17]),
        updated_at=_require_dt(row[18]),
        setting=row[19] or "unknown",
        timing=row[20] or "exact",
        extraction_input_hash=row[21],
        embedding_input_hash=row[22],
    )


def _parse_dt(value: str | None) -> datetime | None:
    """Parse an optional stored ISO timestamp, preserving its offset."""
    return datetime.fromisoformat(value) if value else None


def _require_dt(value: str) -> datetime:
    """Parse a stored ISO timestamp that must be present."""
    return datetime.fromisoformat(value)


def save_events(
    events: list[Event],
    db_path: Path | str,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
) -> None:
    """Insert or update events, with their tags, vectors and provenance.

    A tag's vector is a pure function of its text and the embedding model, so
    it is stored once in `tag_embeddings` rather than once per event that uses
    the tag — measured, 5,289 tag instances share 1,258 distinct vectors. The
    weight stays on `event_tags`, because centrality *is* per event.

    Args:
        events: Events to persist. Existing rows are replaced by event_id.
        db_path: Path to the SQLite database.
        embedding_model: Names which model produced the vectors, so changing
            embedder adds rows rather than silently invalidating the old ones.
    """
    if not events:
        return

    conn = connect(db_path)
    try:
        write_events(conn, events, embedding_model)
        conn.commit()
    finally:
        conn.close()


def write_events(
    conn: sqlite3.Connection,
    events: list[Event],
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
) -> None:
    """Write events on an existing connection, without committing.

    Split out so a caller can group this with other writes in one transaction —
    reconcile deletes superseded events and saves their replacements, and a
    crash between the two must leave neither applied.

    Args:
        conn: An open connection, from `connect`. The caller owns the commit.
        events: Events to persist. Existing rows are replaced by event_id.
        embedding_model: Names which model produced the vectors.

    Raises:
        ValueError: If an event's tag vectors cannot be paired with its tags.
    """
    if not events:
        return

    for event in events:
        validate_tag_vectors(event)

    placeholders = ", ".join("?" * len(EVENT_COLUMNS.split(", ")))
    conn.executemany(
        f"INSERT OR REPLACE INTO events ({EVENT_COLUMNS}) VALUES ({placeholders})",
        [event_to_row(e) for e in events],
    )

    ids = [(e.event_id,) for e in events]
    conn.executemany("DELETE FROM event_tags WHERE event_id = ?", ids)
    conn.executemany("DELETE FROM event_source_candidates WHERE event_id = ?", ids)

    tag_rows = [
        (event.event_id, position, tag.text, tag.weight)
        for event in events
        for position, tag in enumerate(event.tags)
    ]
    conn.executemany(
        "INSERT INTO event_tags (event_id, position, tag, weight) VALUES (?, ?, ?, ?)",
        tag_rows,
    )

    vector_rows = [
        (tag.text, embedding_model, vector, event.updated_at.isoformat())
        for event in events
        for tag, vector in zip(event.tags, event.tag_embeddings)
    ]
    conn.executemany(
        "INSERT OR REPLACE INTO tag_embeddings (tag, model, embedding, created_at) "
        "VALUES (?, ?, ?, ?)",
        vector_rows,
    )

    conn.executemany(
        "INSERT OR IGNORE INTO event_source_candidates (event_id, candidate_id) VALUES (?, ?)",
        [
            (event.event_id, candidate_id)
            for event in events
            for candidate_id in event.source_event_candidates
        ],
    )


def delete_events(event_ids: list[str], db_path: Path | str) -> None:
    """Remove events superseded by a merge.

    Args:
        event_ids: Events to delete. An empty list is a no-op — nothing was
            superseded, which is never the same as "clear the table".
        db_path: Path to the SQLite database.
    """
    if not event_ids:
        return

    conn = connect(db_path)
    try:
        conn.executemany("DELETE FROM events WHERE id = ?", [(i,) for i in event_ids])
        conn.commit()
    finally:
        conn.close()


def load_events(
    db_path: Path | str, embedding_model: str = DEFAULT_EMBEDDING_MODEL
) -> list[Event]:
    """Load all persisted events, with tags, vectors and provenance reattached.

    Args:
        db_path: Path to the SQLite database.
        embedding_model: Which model's vectors to attach. A tag embedded by a
            different model is left without one rather than silently mixed.

    Returns:
        Events with tags, embeddings, and enrichment restored. `similarity` is
        not stored on the event — it is owned by `event_scores`.
    """
    conn = connect(db_path)
    try:
        rows = conn.execute(f"SELECT {EVENT_COLUMNS} FROM events").fetchall()

        # Three bulk reads rather than a query per event: the batch loads every
        # stored event on each run, so per-event round trips would dominate.
        tag_rows = conn.execute(
            "SELECT t.event_id, t.tag, t.weight, v.embedding "
            "FROM event_tags t "
            "LEFT JOIN tag_embeddings v ON v.tag = t.tag AND v.model = ? "
            "ORDER BY t.event_id, t.position",
            (embedding_model,),
        ).fetchall()
        candidate_rows = conn.execute(
            "SELECT event_id, candidate_id FROM event_source_candidates"
        ).fetchall()
    finally:
        conn.close()

    tags_by_event: dict[str, list[Tag]] = {}
    vectors_by_event: dict[str, list[bytes]] = {}
    for event_id, text, weight, embedding in tag_rows:
        tags_by_event.setdefault(event_id, []).append(Tag(text=text, weight=weight))
        if embedding is not None:
            vectors_by_event.setdefault(event_id, []).append(embedding)

    candidates_by_event: dict[str, list[str]] = {}
    for event_id, candidate_id in candidate_rows:
        candidates_by_event.setdefault(event_id, []).append(candidate_id)

    events = []
    for row in rows:
        event_id = row[0]
        tags = tags_by_event.get(event_id, [])
        vectors = vectors_by_event.get(event_id, [])
        # A partially embedded event is treated as unembedded: a positional
        # pairing of unequal lists is exactly the misalignment this schema
        # exists to make impossible.
        if len(vectors) != len(tags):
            vectors = []
        events.append(row_to_event(row, tags, vectors, candidates_by_event.get(event_id, [])))
    return events


def load_tag_embeddings(
    db_path: Path | str, embedding_model: str = DEFAULT_EMBEDDING_MODEL
) -> dict[str, bytes]:
    """Every tag vector already computed by this model, keyed by tag text.

    A vector is a pure function of its text and the model, so a tag embedded on
    any previous night never needs embedding again. Seeds the embedding stage's
    memo, which otherwise starts empty every run and re-embeds the whole corpus.

    Args:
        db_path: Path to the SQLite database.
        embedding_model: Only vectors from this model are returned; mixing
            models in one space would be meaningless.

    Returns:
        Mapping of tag text to its packed float32 vector.
    """
    conn = connect(db_path)
    try:
        rows = conn.execute(
            "SELECT tag, embedding FROM tag_embeddings WHERE model = ?",
            (embedding_model,),
        ).fetchall()
    finally:
        conn.close()

    return {tag: embedding for tag, embedding in rows}
