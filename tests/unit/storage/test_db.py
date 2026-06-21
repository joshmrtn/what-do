import sqlite3

import pytest


def test_all_tables_exist(tmp_path):
    from src.storage.db import init_db

    init_db(db_path=tmp_path / "test.db")
    conn = sqlite3.connect(tmp_path / "test.db")
    cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = {row[0] for row in cursor.fetchall()}
    conn.close()

    expected = {
        "venues",
        "candidate_entities",
        "event_candidates",
        "events",
        "recommendations",
        "preference_embeddings_cache",
        "weather_cache",
        "run_history",
        "feedback",
        "blocklist",
    }
    assert expected.issubset(tables)


def test_events_table_has_blob_columns(tmp_path):
    from src.storage.db import init_db

    init_db(db_path=tmp_path / "test.db")
    conn = sqlite3.connect(tmp_path / "test.db")
    cursor = conn.execute("PRAGMA table_info(events)")
    columns = {row[1]: row[2] for row in cursor.fetchall()}
    conn.close()

    assert columns.get("tag_embeddings") == "BLOB"
    assert columns.get("summary_embedding") == "BLOB"


def test_recommendations_table_columns(tmp_path):
    from src.storage.db import init_db

    init_db(db_path=tmp_path / "test.db")
    conn = sqlite3.connect(tmp_path / "test.db")
    cursor = conn.execute("PRAGMA table_info(recommendations)")
    columns = {row[1]: row[2] for row in cursor.fetchall()}
    conn.close()

    assert "score" in columns
    assert columns["score"] == "REAL"
    assert "reasons" in columns
    assert "tier" in columns
    assert "match" in columns


def test_init_db_idempotent(tmp_path):
    from src.storage.db import init_db

    db_path = tmp_path / "test.db"
    init_db(db_path=db_path)
    init_db(db_path=db_path)  # must not raise or duplicate tables
