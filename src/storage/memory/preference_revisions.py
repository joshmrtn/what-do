"""In-memory `PreferenceRevisionRepository` — the single official fake."""

from __future__ import annotations

import uuid
from dataclasses import replace

from src.models.preference_revision import PreferenceRevision


class InMemoryPreferenceRevisionRepository:
    """Holds revisions in a dict, keyed by the id it minted for each."""

    def __init__(self) -> None:
        self._by_id: dict[str, PreferenceRevision] = {}
        self._by_hash: dict[str, str] = {}

    def record(self, revision: PreferenceRevision) -> str:
        """Store a revision, or recognise one already held, returning its id."""
        existing = self._by_hash.get(revision.content_hash)
        if existing is not None:
            return existing

        revision_id = str(uuid.uuid4())
        self._by_id[revision_id] = revision
        self._by_hash[revision.content_hash] = revision_id
        return revision_id

    def get(self, revision_id: str) -> PreferenceRevision | None:
        """One revision by id, with its lines in file order, or None."""
        held = self._by_id.get(revision_id)
        return None if held is None else _ordered(held)

    def latest(self) -> PreferenceRevision | None:
        """The most recently captured revision, or None if there are none."""
        if not self._by_id:
            return None
        return _ordered(
            max(self._by_id.values(), key=lambda revision: revision.captured_at)
        )


def _ordered(revision: PreferenceRevision) -> PreferenceRevision:
    """The revision with its lines as they sat in their files.

    The SQLite repository orders on read, so this must too. Returning insertion
    order instead is the drift a contract suite exists to catch — and did.
    """
    return replace(
        revision,
        lines=sorted(revision.lines, key=lambda line: (line.file_name, line.position)),
    )
