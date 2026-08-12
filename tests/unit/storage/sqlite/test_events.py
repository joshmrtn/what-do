"""Behaviour only a database can have.

Deliberately outside the shared contract suite: foreign keys, cascades and
"which rows were actually written" are properties of SQLite, not of storage in
general. Asserting them for every implementation would force the in-memory fake
to reimplement referential integrity, which is how a fake grows its own bugs.
"""

from __future__ import annotations

import zoneinfo
from datetime import datetime, timezone

import pytest

from src.models.event import Event
from src.models.tag import Tag
from src.storage.sqlite.connection import connect, init_db
from src.storage.sqlite.events import SqliteEventRepository
from src.utils.vectors import encode_vector

_TZ = zoneinfo.ZoneInfo("America/New_York")
_NOW = datetime(2026, 6, 15, 12, 0, tzinfo=timezone.utc)


@pytest.fixture
def repo(tmp_path):
    path = tmp_path / "events.db"
    init_db(path)
    return SqliteEventRepository(path)


def _event(event_id="e1", tags=None, vectors=None, **kwargs) -> Event:
    defaults = dict(
        source_event_candidates=["c1"],
        source_type="apify",
        created_at=_NOW,
        updated_at=_NOW,
        title="Karaoke Night",
        start_time=datetime(2026, 8, 26, 20, 30, tzinfo=_TZ),
    )
    defaults.update(kwargs)
    event = Event(event_id=event_id, tags=tags or [], **defaults)
    event.tag_embeddings = vectors or []
    return event


def _count(path, table, event_id="e1") -> int:
    conn = connect(path)
    try:
        return conn.execute(
            f"SELECT count(*) FROM {table} WHERE event_id = ?", (event_id,)
        ).fetchone()[0]
    finally:
        conn.close()


def test_delete_cascades_to_tags_candidates_and_images(repo):
    """A deleted event must not leave rows pointing at nothing."""
    repo.save([_event(tags=[Tag("karaoke", 1.0)], vectors=[encode_vector([1.0, 2.0])])])
    conn = connect(repo._db_path)
    conn.execute("INSERT INTO event_images (event_id, bytes) VALUES ('e1', X'00')")
    conn.commit()
    conn.close()

    repo.delete(["e1"])

    assert _count(repo._db_path, "event_tags") == 0
    assert _count(repo._db_path, "event_source_candidates") == 0
    assert _count(repo._db_path, "event_images") == 0


def test_delete_is_refused_while_feedback_references_the_event(repo):
    """Feedback deliberately does not cascade.

    Merging a duplicate must never silently discard a rating the user gave, so
    the reference refuses the delete instead of following it. Without this test
    someone tidying the schema makes cascades consistent and the rating is gone.
    """
    import sqlite3

    repo.save([_event()])
    conn = connect(repo._db_path)
    conn.execute(
        "INSERT INTO feedback (id, event_id, rating, submitted_at) VALUES (?, ?, ?, ?)",
        ("f1", "e1", "up", _NOW.isoformat()),
    )
    conn.commit()
    conn.close()

    with pytest.raises(sqlite3.IntegrityError):
        repo.delete(["e1"])

    assert [e.event_id for e in repo.load_all()] == ["e1"]


def test_save_one_writes_only_its_own_row(repo):
    """The claim that removes the reason extraction batches its saves.

    Counted with a trigger rather than by comparing row contents: passing the
    whole corpus to `save` rewrites every row to identical values, which no
    assertion on the data itself can distinguish from writing one.
    """
    repo.save([_event("first"), _event("second")])

    conn = connect(repo._db_path)
    conn.execute("CREATE TABLE writes (id TEXT)")
    conn.execute(
        "CREATE TRIGGER count_writes AFTER INSERT ON events "
        "BEGIN INSERT INTO writes VALUES (NEW.id); END"
    )
    conn.commit()
    conn.close()

    repo.save_one(_event("third"))

    conn = connect(repo._db_path)
    written = [row[0] for row in conn.execute("SELECT id FROM writes")]
    conn.close()

    assert written == ["third"]


def test_a_tag_shared_by_two_events_is_stored_once(repo):
    """A vector is a function of (tag, model), so it is not stored per event."""
    vector = encode_vector([1.0, 2.0, 3.0])
    repo.save(
        [
            _event("e1", tags=[Tag("karaoke", 1.0)], vectors=[vector]),
            _event("e2", tags=[Tag("karaoke", 0.4)], vectors=[vector]),
        ]
    )

    conn = connect(repo._db_path)
    try:
        rows = conn.execute("SELECT tag, count(*) FROM tag_embeddings GROUP BY tag").fetchall()
        weights = conn.execute(
            "SELECT event_id, weight FROM event_tags ORDER BY event_id"
        ).fetchall()
    finally:
        conn.close()

    assert rows == [("karaoke", 1)]
    # The weight is per event even though the vector is not.
    assert weights == [("e1", 1.0), ("e2", 0.4)]


def test_replace_saves_the_replacements_and_drops_the_superseded(repo):
    repo.save([_event("stale")])

    repo.replace(["stale"], [_event("merged")])

    assert [e.event_id for e in repo.load_all()] == ["merged"]


def test_replace_applies_neither_half_when_the_save_fails(monkeypatch, repo):
    """Reconcile's delete and save were separate transactions.

    A crash between them removed rows whose replacements never arrived, and
    nothing would restore them on the next run.
    """
    repo.save([_event("stale")])

    def boom(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr("src.storage.sqlite.events.write_events", boom)

    with pytest.raises(RuntimeError, match="boom"):
        repo.replace(["stale"], [_event("merged")])

    assert [e.event_id for e in repo.load_all()] == ["stale"]
