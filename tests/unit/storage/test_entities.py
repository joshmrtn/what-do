from __future__ import annotations

import sqlite3

import pytest

from src.storage.db import init_db
from src.storage.entities import load_active_handles


@pytest.fixture
def db(tmp_path):
    path = tmp_path / "batch.db"
    init_db(path)
    return path


def _insert(db, handle: str, state: str) -> None:
    conn = sqlite3.connect(db)
    conn.execute(
        "INSERT INTO candidate_entities (id, handle, state, created_at, updated_at) "
        "VALUES (?, ?, ?, '2026-06-15T12:00:00+00:00', '2026-06-15T12:00:00+00:00')",
        (handle, handle, state),
    )
    conn.commit()
    conn.close()


def test_active_handles_are_returned(db):
    _insert(db, "@jazzclub", "active")

    assert load_active_handles(db) == ["@jazzclub"]


def test_probationary_handles_are_not_returned(db):
    """Promotion is what makes a discovered handle worth spending a fetch on."""
    _insert(db, "@maybe", "probationary")

    assert load_active_handles(db) == []


def test_handles_come_back_in_a_stable_order(db):
    for handle in ["@zed", "@alpha", "@mid"]:
        _insert(db, handle, "active")

    assert load_active_handles(db) == ["@alpha", "@mid", "@zed"]


def test_an_empty_table_returns_nothing(db):
    assert load_active_handles(db) == []


def test_an_uninitialised_database_returns_nothing(tmp_path):
    """sqlite3.connect would create a zero-byte file; has_schema is the real check."""
    assert load_active_handles(tmp_path / "never-made.db") == []
