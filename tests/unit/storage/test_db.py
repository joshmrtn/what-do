import sqlite3

import pytest

from src.storage.db import has_schema, init_db


def test_all_tables_exist(tmp_path):

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
    }
    assert expected.issubset(tables)


def test_events_table_has_blob_columns(tmp_path):

    init_db(db_path=tmp_path / "test.db")
    conn = sqlite3.connect(tmp_path / "test.db")
    cursor = conn.execute("PRAGMA table_info(events)")
    columns = {row[1]: row[2] for row in cursor.fetchall()}
    conn.close()

    assert columns.get("tag_embeddings") == "BLOB"
    assert columns.get("summary_embedding") == "BLOB"


def test_recommendations_table_columns(tmp_path):

    init_db(db_path=tmp_path / "test.db")
    conn = sqlite3.connect(tmp_path / "test.db")
    cursor = conn.execute("PRAGMA table_info(recommendations)")
    columns = {row[1]: row[2] for row in cursor.fetchall()}
    conn.close()

    for score_column in ("base_score", "weather_adjustment", "tag_confidence", "final_score"):
        assert columns.get(score_column) == "REAL"
    assert columns.get("rank") == "INTEGER"
    assert "reasons" in columns
    assert "tier" in columns
    assert "match" in columns


def test_init_db_idempotent(tmp_path):

    db_path = tmp_path / "test.db"
    init_db(db_path=db_path)
    init_db(db_path=db_path)  # must not raise or duplicate tables


def test_event_candidates_has_raw_published_at(tmp_path):

    init_db(db_path=tmp_path / "test.db")
    conn = sqlite3.connect(tmp_path / "test.db")
    cursor = conn.execute("PRAGMA table_info(event_candidates)")
    columns = {row[1] for row in cursor.fetchall()}
    conn.close()

    assert "raw_published_at" in columns


def test_candidate_entities_has_depth_and_mention_sources(tmp_path):

    init_db(db_path=tmp_path / "test.db")
    conn = sqlite3.connect(tmp_path / "test.db")
    cursor = conn.execute("PRAGMA table_info(candidate_entities)")
    columns = {row[1] for row in cursor.fetchall()}
    conn.close()

    assert "depth" in columns
    assert "mention_sources" in columns


def test_events_table_has_image_bytes_blob(tmp_path):

    init_db(db_path=tmp_path / "test.db")
    conn = sqlite3.connect(tmp_path / "test.db")
    cursor = conn.execute("PRAGMA table_info(events)")
    columns = {row[1]: row[2] for row in cursor.fetchall()}
    conn.close()

    assert "image_bytes" in columns
    assert columns["image_bytes"] == "BLOB"


def test_has_schema_is_false_when_the_file_does_not_exist(tmp_path):
    assert has_schema(tmp_path / "never_created.db") is False


def test_has_schema_is_false_for_an_empty_file(tmp_path):
    """sqlite3.connect creates a zero-byte file, so existence proves nothing."""
    path = tmp_path / "touched.db"
    sqlite3.connect(path).close()

    assert has_schema(path) is False


def test_has_schema_is_true_after_init(tmp_path):
    path = tmp_path / "real.db"
    init_db(path)

    assert has_schema(path) is True
