"""SQLite EventRepository.

Wraps the existing module-level persistence functions rather than restating
their SQL. They already hold the row mapping, the tag/vector split and the
provenance writes, all verified against the live corpus; duplicating that here
to satisfy a shape would be two implementations of the same thing, drifting.

What this class adds is the boundary: callers hold a repository and no longer
thread `db_path` through stages that have no other use for it.
"""

from __future__ import annotations

from pathlib import Path

from src.config import DEFAULT_EMBEDDING_MODEL
from src.models.event import Event
from src.storage.db import connect, transaction
from src.storage.events import (
    delete_events,
    load_events,
    load_tag_embeddings,
    save_events,
    write_events,
)


class SqliteEventRepository:
    """Events in SQLite, with tags, vectors and provenance."""

    def __init__(
        self, db_path: Path | str, embedding_model: str = DEFAULT_EMBEDDING_MODEL
    ) -> None:
        """Args:
        db_path: Path to the SQLite database.
        embedding_model: Which model's vectors this repository reads and
            writes. A tag embedded by another model is left unattached
            rather than silently mixed into the same space.
        """
        self._db_path = db_path
        self._embedding_model = embedding_model

    def save(self, events: list[Event]) -> None:
        """Insert or replace events. See `EventRepository.save`."""
        save_events(events, self._db_path, self._embedding_model)

    def save_one(self, event: Event) -> None:
        """Persist a single event. See `EventRepository.save_one`."""
        save_events([event], self._db_path, self._embedding_model)

    def load_all(self) -> list[Event]:
        """Every stored event, with tags, vectors and provenance reattached."""
        return load_events(self._db_path, self._embedding_model)

    def delete(self, event_ids: list[str]) -> None:
        """Remove events superseded by a merge."""
        delete_events(event_ids, self._db_path)

    def tag_embeddings(self) -> dict[str, bytes]:
        """Every tag vector already computed by this model, keyed by tag text."""
        return load_tag_embeddings(self._db_path, self._embedding_model)

    def replace(self, stale_ids: list[str], events: list[Event]) -> None:
        """Delete superseded events and save their replacements, atomically.

        Reconcile's two halves were separate transactions, so a crash between
        them removed rows whose replacements never arrived and nothing would
        restore them.

        Args:
            stale_ids: Events superseded by a merge.
            events: Events to persist in their place.
        """
        conn = connect(self._db_path)
        try:
            with transaction(conn):
                if stale_ids:
                    conn.executemany(
                        "DELETE FROM events WHERE id = ?", [(i,) for i in stale_ids]
                    )
                write_events(conn, events, self._embedding_model)
        finally:
            conn.close()
