"""In-memory EventRepository — the official fake.

Exists so tests can substitute storage without SQLite *and* without inventing a
one-off stub per test module. Hand-written stubs drift from the real contract
silently, and the suite stays green while production breaks; one shared
implementation running the same contract tests as SQLite cannot.

It is a store, not a database: no foreign keys, no cascades. Behaviour that
depends on those is tested against SQLite, because a fake that reimplements
referential integrity is just a worse database with its own bugs.
"""

from __future__ import annotations

import copy

from src.config import DEFAULT_EMBEDDING_MODEL
from src.models.event import Event
from src.storage.events import validate_tag_vectors


class InMemoryEventRepository:
    """Events held in a dict, keyed by event id."""

    def __init__(self, embedding_model: str = DEFAULT_EMBEDDING_MODEL) -> None:
        """Args:
        embedding_model: Named for parity with the SQLite repository, which
            keys vectors by it. Held so the two construct alike.
        """
        self._events: dict[str, Event] = {}
        self._embedding_model = embedding_model

    def save(self, events: list[Event]) -> None:
        """Insert or replace events. See `EventRepository.save`."""
        for event in events:
            validate_tag_vectors(event)
        for event in events:
            # Copied on the way in and out, so a caller mutating an event it
            # saved cannot reach back into the store. SQLite gets that for free
            # by serialising; without it this fake would be more permissive than
            # the thing it stands in for, and round-trip tests would pass by
            # comparing an object with itself.
            stored = copy.deepcopy(event)
            # `similarity` has no column, by decision: it is derived, cheap to
            # recompute, and owned by `event_scores`. Keeping it here would make
            # this store remember something the real one cannot, which is the
            # one way an in-memory implementation quietly stops being a
            # substitute for the thing it doubles.
            stored.similarity = None
            self._events[event.event_id] = stored

    def save_one(self, event: Event) -> None:
        """Persist a single event. See `EventRepository.save_one`."""
        self.save([event])

    def load_all(self) -> list[Event]:
        """Every stored event, in insertion order."""
        return [copy.deepcopy(event) for event in self._events.values()]

    def delete(self, event_ids: list[str]) -> None:
        """Remove events by id, ignoring ids that are not present."""
        for event_id in event_ids:
            self._events.pop(event_id, None)

    def replace(self, stale_ids: list[str], events: list[Event]) -> None:
        """Delete superseded events and save their replacements.

        Validation runs before anything is removed, so a rejected event cannot
        leave the store having dropped the rows it was meant to supersede — the
        atomicity SQLite gets from a transaction.
        """
        for event in events:
            validate_tag_vectors(event)
        self.delete(stale_ids)
        self.save(events)

    def tag_embeddings(self) -> dict[str, bytes]:
        """Every tag vector held, keyed by tag text."""
        vectors: dict[str, bytes] = {}
        for event in self._events.values():
            for tag, vector in zip(event.tags, event.tag_embeddings):
                vectors[tag.text] = vector
        return vectors
