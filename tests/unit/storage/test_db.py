import sqlite3

import pytest

from src.storage.db import connect, has_schema, init_db, transaction


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
        "rankings",
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

    # Tag vectors moved to `tag_embeddings`, keyed by (tag, model): a vector is
    # a property of the text, not of the event that happens to use it.
    assert columns.get("summary_embedding") == "BLOB"
    assert "tag_embeddings" not in columns


def test_rankings_table_columns(tmp_path):

    init_db(db_path=tmp_path / "test.db")
    conn = sqlite3.connect(tmp_path / "test.db")
    cursor = conn.execute("PRAGMA table_info(rankings)")
    columns = {row[1]: row[2] for row in cursor.fetchall()}
    conn.close()

    # `rankings` is the placement alone — what depends on the night. The verdict
    # on the event lives in `event_scores` and its breakdown in `score_reasons`,
    # so both survive for events that were scored but fell outside the window.
    for placement_column in ("weather_adjustment", "final_score"):
        assert columns.get(placement_column) == "REAL"
    assert columns.get("rank") == "INTEGER"
    # The verdict and its breakdown stay on `event_scores` / `score_reasons`,
    # so both survive for events scored but outside the ranked window.
    assert "base_score" not in columns
    assert "reasons" not in columns
    assert "match" not in columns
    # tag_confidence is a pure function of the event's own tags, so it belongs
    # with the verdict, not the placement.
    assert "tag_confidence" not in columns


def test_event_scores_carries_the_confidence_and_both_components(tmp_path):
    init_db(db_path=tmp_path / "test.db")
    conn = sqlite3.connect(tmp_path / "test.db")
    columns = {row[1]: row[2] for row in conn.execute("PRAGMA table_info(event_scores)")}
    conn.close()

    for column in ("tag_score", "summary_score", "base_score", "tag_confidence"):
        assert columns.get(column) == "REAL", column
    assert columns.get("match") == "TEXT"


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


def test_image_bytes_live_outside_the_hot_events_row(tmp_path):
    """A blob column widens every scan of the table the batch reads most."""
    init_db(db_path=tmp_path / "test.db")
    conn = sqlite3.connect(tmp_path / "test.db")
    events = {row[1]: row[2] for row in conn.execute("PRAGMA table_info(events)")}
    images = {row[1]: row[2] for row in conn.execute("PRAGMA table_info(event_images)")}
    conn.close()

    assert "image_bytes" not in events
    assert images["bytes"] == "BLOB"


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


def _venue(conn, venue_id):
    """Insert one venue, the smallest write this database accepts."""
    conn.execute(
        "INSERT INTO venues (id, name, discovered_at) VALUES (?, ?, ?)",
        (venue_id, f"Venue {venue_id}", "2026-08-10T00:00:00+00:00"),
    )


def _venue_ids(path):
    """Read committed venue ids through a fresh connection.

    Proves durability — a second connection sees only committed state.
    """
    conn = connect(path)
    try:
        return {row[0] for row in conn.execute("SELECT id FROM venues")}
    finally:
        conn.close()


def _assert_rolled_back(conn, path):
    """Assert a failed block left nothing behind, on the only probe that proves it.

    Closing a connection discards an open transaction by itself, so checking a
    fresh connection *after* closing passes whether or not the block rolled
    back. The writes have to be shown absent on the connection that made them,
    while it is still open — uncommitted work is visible there, so a missing
    rollback shows up as a leftover row rather than as silence.
    """
    assert conn.execute("SELECT id FROM venues").fetchall() == []
    assert conn.in_transaction is False, "the savepoint is still open"
    conn.close()
    assert _venue_ids(path) == set()


def test_transaction_commits_on_clean_exit(tmp_path):
    path = tmp_path / "commit.db"
    init_db(path)
    conn = connect(path)

    with transaction(conn):
        _venue(conn, "kept")
    conn.close()

    assert _venue_ids(path) == {"kept"}


def test_transaction_rolls_back_on_exception(tmp_path):
    """The case that matters: reconcile deletes then saves, and a crash between
    the two must leave neither applied."""
    path = tmp_path / "rollback.db"
    init_db(path)
    conn = connect(path)

    with pytest.raises(RuntimeError, match="boom"):
        with transaction(conn):
            _venue(conn, "discarded")
            raise RuntimeError("boom")

    _assert_rolled_back(conn, path)


def test_transaction_rolls_back_every_write_in_the_block(tmp_path):
    """A partial rollback would be worse than none: it looks like success."""
    path = tmp_path / "partial.db"
    init_db(path)
    conn = connect(path)

    with pytest.raises(RuntimeError):
        with transaction(conn):
            _venue(conn, "first")
            _venue(conn, "second")
            raise RuntimeError("boom")

    _assert_rolled_back(conn, path)


def test_nested_transaction_commits_with_the_outermost_block(tmp_path):
    """SQLite has no true nested transactions, so an inner block must not commit
    early — the outer block still owns the decision."""
    path = tmp_path / "nested_commit.db"
    init_db(path)
    conn = connect(path)

    with transaction(conn):
        _venue(conn, "outer")
        with transaction(conn):
            _venue(conn, "inner")
        assert _venue_ids(path) == set(), "inner block committed before the outer finished"
    conn.close()

    assert _venue_ids(path) == {"outer", "inner"}


def test_nested_transaction_outer_rollback_discards_inner_work(tmp_path):
    path = tmp_path / "nested_rollback.db"
    init_db(path)
    conn = connect(path)

    with pytest.raises(RuntimeError):
        with transaction(conn):
            with transaction(conn):
                _venue(conn, "inner")
            raise RuntimeError("boom")

    _assert_rolled_back(conn, path)


def test_inner_rollback_leaves_the_outer_block_free_to_commit(tmp_path):
    """The reason this is built on savepoints rather than BEGIN.

    A failed inner block undoes only its own writes; the enclosing block keeps
    its work and still commits. With BEGIN the whole transaction would be dead.
    """
    path = tmp_path / "inner_rollback.db"
    init_db(path)
    conn = connect(path)

    with transaction(conn):
        _venue(conn, "outer")
        with pytest.raises(RuntimeError):
            with transaction(conn):
                _venue(conn, "inner")
                raise RuntimeError("boom")
        _venue(conn, "after")
    conn.close()

    assert _venue_ids(path) == {"outer", "after"}


def test_transaction_rolls_back_when_the_process_is_interrupted(tmp_path):
    """A batch writes for hours and can be killed mid-block.

    KeyboardInterrupt derives from BaseException, not Exception, so a bare
    `except Exception` would commit a half-written block on Ctrl-C.
    """
    path = tmp_path / "interrupted.db"
    init_db(path)
    conn = connect(path)

    with pytest.raises(KeyboardInterrupt):
        with transaction(conn):
            _venue(conn, "half_written")
            raise KeyboardInterrupt

    _assert_rolled_back(conn, path)


def test_rollback_undoes_cascaded_deletes_too(tmp_path):
    """Reconcile deletes events then saves; the delete cascades to child rows.

    Rolling back the delete without its cascade would leave the parent restored
    and its children gone — worse than either outcome alone.
    """
    path = tmp_path / "cascade.db"
    init_db(path)
    conn = connect(path)
    _venue(conn, "venue")
    conn.execute("INSERT INTO venue_handles (venue_id, handle) VALUES (?, ?)", ("venue", "@a"))
    conn.commit()

    with pytest.raises(RuntimeError):
        with transaction(conn):
            conn.execute("DELETE FROM venues WHERE id = 'venue'")
            assert conn.execute("SELECT count(*) FROM venue_handles").fetchone()[0] == 0
            raise RuntimeError("boom")

    handles = conn.execute("SELECT handle FROM venue_handles").fetchall()
    conn.close()

    assert _venue_ids(path) == {"venue"}
    assert handles == [("@a",)]


def test_a_reader_is_not_blocked_by_an_open_write_transaction(tmp_path):
    """The failure that prompted this refactor: `what-do` reading mid-batch.

    Under WAL a reader sees the last committed state instead of blocking, so an
    hours-long write block never makes the CLI fail.
    """
    path = tmp_path / "concurrent.db"
    init_db(path)
    writer = connect(path)
    _venue(writer, "committed")
    writer.commit()

    with transaction(writer):
        _venue(writer, "uncommitted")
        visible = _venue_ids(path)
    writer.close()

    assert visible == {"committed"}


def test_transaction_is_reusable_after_a_rollback(tmp_path):
    """A failed stage must not poison the connection for the stages after it."""
    path = tmp_path / "reuse.db"
    init_db(path)
    conn = connect(path)

    with pytest.raises(RuntimeError):
        with transaction(conn):
            _venue(conn, "discarded")
            raise RuntimeError("boom")

    with transaction(conn):
        _venue(conn, "kept")
    conn.close()

    assert _venue_ids(path) == {"kept"}
