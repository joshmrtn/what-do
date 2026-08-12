"""In-memory `EntityRepository` — the single official fake.

Accumulation is the behaviour that matters here, so this implements it rather
than approximating it: a fake that replaces where the real one increments would
pass its own tests and lie to every caller.
"""

from __future__ import annotations

import uuid
from dataclasses import replace
from datetime import datetime
from typing import Callable

from src.models.candidate_entity import ACTIVE, PROBATIONARY, CandidateEntity


class InMemoryEntityRepository:
    """Holds candidate entities in a dict keyed by handle."""

    def __init__(self) -> None:
        self._by_handle: dict[str, CandidateEntity] = {}

    def active_handles(self) -> list[str]:
        """Every handle currently active for ingestion, alphabetically."""
        return sorted(
            entity.handle
            for entity in self._by_handle.values()
            if entity.state == ACTIVE
        )

    def mark_seeds_active(self, handles: list[str], *, now: datetime) -> None:
        """Upsert seed handles as active at depth 0, keeping any counters."""
        for handle in handles:
            existing = self._by_handle.get(handle)
            if existing is not None:
                self._by_handle[handle] = replace(
                    existing, state=ACTIVE, depth=0, updated_at=now
                )
            else:
                self._by_handle[handle] = CandidateEntity(
                    entity_id=str(uuid.uuid4()),
                    handle=handle,
                    state=ACTIVE,
                    depth=0,
                    created_at=now,
                    updated_at=now,
                )

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
        existing = self._by_handle.get(handle)
        if existing is None:
            self._by_handle[handle] = CandidateEntity(
                entity_id=str(uuid.uuid4()),
                handle=handle,
                state=PROBATIONARY,
                depth=depth,
                mention_count=1,
                mention_sources=[source_handle],
                discovery_context=context,
                created_at=now,
                updated_at=now,
            )
            return

        if source_handle in existing.mention_sources:
            return

        self._by_handle[handle] = replace(
            existing,
            mention_count=existing.mention_count + 1,
            mention_sources=[*existing.mention_sources, source_handle],
            # First one wins, matching COALESCE.
            discovery_context=existing.discovery_context or context,
            updated_at=now,
        )

    def by_handle(self, handle: str) -> CandidateEntity | None:
        """One entity by its handle, or None if it has never been seen."""
        return self._by_handle.get(handle)

    def unclassified(self) -> list[CandidateEntity]:
        """Probationary handles disambiguation has not yet judged."""
        return self._matching(
            lambda e: e.state == PROBATIONARY and e.llm_classification is None
        )

    def classify(
        self, entity_id: str, *, classification: str, state: str, now: datetime
    ) -> None:
        """Record what disambiguation decided, and the state that follows."""
        for handle, entity in self._by_handle.items():
            if entity.entity_id == entity_id:
                self._by_handle[handle] = replace(
                    entity,
                    llm_classification=classification,
                    state=state,
                    updated_at=now,
                )
                return

    def awaiting_promotion(self) -> list[CandidateEntity]:
        """Probationary handles classified as venues, with their evidence."""
        return self._matching(
            lambda e: e.state == PROBATIONARY and e.llm_classification == "venue"
        )

    def activate(self, entity_id: str, *, now: datetime) -> None:
        """Promote a handle to `active`."""
        for handle, entity in self._by_handle.items():
            if entity.entity_id == entity_id:
                self._by_handle[handle] = replace(entity, state=ACTIVE, updated_at=now)
                return

    def _matching(
        self, predicate: Callable[[CandidateEntity], bool]
    ) -> list[CandidateEntity]:
        return sorted(
            (e for e in self._by_handle.values() if predicate(e)),
            key=lambda e: e.handle,
        )
