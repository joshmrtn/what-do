"""Unit tests for the pre-re-key database snapshot."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from src.storage.snapshot import snapshot_database
from src.storage.sqlite.connection import connect, init_db

NOW = datetime(2026, 8, 18, 6, 0, tzinfo=timezone.utc)


@pytest.fixture
def db(tmp_path):
    path = tmp_path / "event_hub.db"
    init_db(path)
    with connect(path) as conn:
        conn.execute(
            "INSERT INTO events (id, title, source_type, created_at, updated_at) "
            "VALUES ('e1', 'A show', 'nsno', ?, ?)",
            (NOW.isoformat(), NOW.isoformat()),
        )
        conn.commit()
    return path


def test_it_writes_a_readable_copy(db):
    taken = snapshot_database(db, reason="latch-nsno", at=NOW)

    with connect(taken) as conn:
        assert conn.execute("SELECT title FROM events").fetchone()[0] == "A show"


def test_the_name_says_why_and_when(db):
    taken = snapshot_database(db, reason="latch-nsno", at=NOW)

    assert "latch-nsno" in taken.name
    assert "20260818" in taken.name


def test_it_lands_beside_the_database_it_protects(db):
    taken = snapshot_database(db, reason="latch-nsno", at=NOW)

    assert taken.parent == db.parent


def test_the_original_is_untouched(db):
    snapshot_database(db, reason="latch-nsno", at=NOW)

    with connect(db) as conn:
        assert conn.execute("SELECT COUNT(*) FROM events").fetchone()[0] == 1


def test_two_snapshots_do_not_collide(db):
    first = snapshot_database(db, reason="latch-a", at=NOW)
    second = snapshot_database(db, reason="latch-b", at=NOW)

    assert first != second
    assert first.exists() and second.exists()


def test_it_is_a_single_file_with_no_sidecars(db):
    """`VACUUM INTO`, not a copy. With WAL on, recent commits live in a sidecar,
    so `cp` captures a torn state — the standing footgun."""
    taken = snapshot_database(db, reason="latch-nsno", at=NOW)

    assert not taken.with_suffix(".db-wal").exists()
    assert not taken.with_suffix(".db-shm").exists()
