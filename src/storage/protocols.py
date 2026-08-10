"""Storage contracts the core depends on.

Core code — stages, services, the scheduler — depends on these Protocols and
never on a concrete store. That is what lets a stage be tested without SQLite,
and what keeps connection handling and row mapping in one place instead of
spread across the modules that happen to need data.
"""

from __future__ import annotations

from typing import Protocol

from src.models.event import Event


class EventRepository(Protocol):
    """Persistence for canonical events, with their tags, vectors and provenance."""

    def save(self, events: list[Event]) -> None:
        """Insert or replace events, replacing each one's tags and provenance.

        Args:
            events: Events to persist. An empty list is a no-op — "nothing to
                save" is never "clear the store".

        Raises:
            ValueError: If an event carries a number of tag vectors that does
                not match its number of tags. Storing the overlap would drop
                the rest and report success.
        """
        ...

    def save_one(self, event: Event) -> None:
        """Persist a single event.

        Exists because extraction costs minutes per event and the batch has to
        checkpoint as it goes. Passing the whole corpus to `save` for each one
        rewrites every row to store one, which is what forced batching before.
        """
        ...

    def load_all(self) -> list[Event]:
        """Every stored event, with tags, vectors and provenance reattached."""
        ...

    def delete(self, event_ids: list[str]) -> None:
        """Remove events superseded by a merge.

        Args:
            event_ids: Events to delete. An empty list is a no-op.
        """
        ...

    def replace(self, stale_ids: list[str], events: list[Event]) -> None:
        """Delete superseded events and save their replacements, atomically.

        Reconcile identifies superseded duplicates hours before the run has
        anything to write in their place. Deleting them at that point opens a
        window across enrichment and extraction where the duplicate is gone and
        the merged winner was never stored; holding one transaction across those
        hours instead would lock the database for the whole batch, which is the
        failure this boundary exists to remove. So the delete travels with the
        save and both take milliseconds.

        Args:
            stale_ids: Events superseded by a merge.
            events: Events to persist in their place.
        """
        ...

    def tag_embeddings(self) -> dict[str, bytes]:
        """Every tag vector already computed, keyed by tag text.

        A vector is a pure function of its text and the embedding model, so a
        tag embedded on a previous night never needs embedding again.
        """
        ...
