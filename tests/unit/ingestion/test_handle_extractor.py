"""Unit tests for HandleExtractor."""

from __future__ import annotations

import io
import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pytest

from src.storage.db import init_db
from src.utils.logging import get_logger


def _make_logger():
    return get_logger("test_extractor", stream=io.StringIO())


@pytest.fixture
def db(tmp_path):
    path = tmp_path / "test.db"
    init_db(db_path=path)
    return path


def _insert_candidate(conn, handle, state="probationary", depth=0, mention_count=0, mention_sources=None):
    conn.execute(
        """INSERT INTO candidate_entities
           (id, handle, state, depth, mention_count, mention_sources, created_at, updated_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            str(uuid.uuid4()),
            handle,
            state,
            depth,
            mention_count,
            json.dumps(mention_sources or []),
            datetime.now(timezone.utc).isoformat(),
            datetime.now(timezone.utc).isoformat(),
        ),
    )
    conn.commit()


def _get_entity(conn, handle):
    row = conn.execute(
        "SELECT state, depth, mention_count, mention_sources FROM candidate_entities WHERE handle = ?",
        (handle,),
    ).fetchone()
    if row is None:
        return None
    return {
        "state": row[0],
        "depth": row[1],
        "mention_count": row[2],
        "mention_sources": json.loads(row[3]) if row[3] else [],
    }


# ---------------------------------------------------------------------------
# Extraction tests
# ---------------------------------------------------------------------------


def test_extracts_single_handle(db):
    from src.ingestion.handle_extractor import HandleExtractor

    extractor = HandleExtractor(db_path=db, max_depth=2, blocklist=[], logger=_make_logger())
    extractor.process("Check out @jazzclub tonight!", source_handle="@seedvenue", source_depth=0)

    conn = sqlite3.connect(db)
    entity = _get_entity(conn, "@jazzclub")
    conn.close()

    assert entity is not None
    assert entity["state"] == "probationary"
    assert entity["depth"] == 1


def test_extracts_multiple_handles(db):
    from src.ingestion.handle_extractor import HandleExtractor

    extractor = HandleExtractor(db_path=db, max_depth=2, blocklist=[], logger=_make_logger())
    extractor.process(
        "Come see @band1 and @band2 at the show!",
        source_handle="@seedvenue",
        source_depth=0,
    )

    conn = sqlite3.connect(db)
    assert _get_entity(conn, "@band1") is not None
    assert _get_entity(conn, "@band2") is not None
    conn.close()


def test_mention_count_incremented_for_existing(db):
    from src.ingestion.handle_extractor import HandleExtractor

    conn = sqlite3.connect(db)
    _insert_candidate(conn, "@jazzclub", mention_count=1, mention_sources=["@other"])

    extractor = HandleExtractor(db_path=db, max_depth=2, blocklist=[], logger=_make_logger())
    extractor.process("Shoutout to @jazzclub!", source_handle="@seedvenue", source_depth=0)

    entity = _get_entity(conn, "@jazzclub")
    conn.close()

    assert entity["mention_count"] == 2
    assert "@seedvenue" in entity["mention_sources"]


def test_same_source_does_not_double_count(db):
    from src.ingestion.handle_extractor import HandleExtractor

    conn = sqlite3.connect(db)
    _insert_candidate(conn, "@jazzclub", mention_count=1, mention_sources=["@seedvenue"])

    extractor = HandleExtractor(db_path=db, max_depth=2, blocklist=[], logger=_make_logger())
    extractor.process("Again @jazzclub!", source_handle="@seedvenue", source_depth=0)

    entity = _get_entity(conn, "@jazzclub")
    conn.close()

    # Count should NOT increment because @seedvenue already in mention_sources
    assert entity["mention_count"] == 1


def test_handle_at_max_depth_not_stored(db):
    from src.ingestion.handle_extractor import HandleExtractor

    extractor = HandleExtractor(db_path=db, max_depth=2, blocklist=[], logger=_make_logger())
    # source_depth=2 means discovered handle would be depth=3, which exceeds max_depth=2
    extractor.process("Visit @deepvenue!", source_handle="@somehandle", source_depth=2)

    conn = sqlite3.connect(db)
    entity = _get_entity(conn, "@deepvenue")
    conn.close()

    assert entity is None


def test_blocklisted_handle_not_stored(db):
    from src.ingestion.handle_extractor import HandleExtractor

    extractor = HandleExtractor(
        db_path=db,
        max_depth=2,
        blocklist=["@badplace"],
        logger=_make_logger(),
    )
    extractor.process("Visit @badplace tonight!", source_handle="@seed", source_depth=0)

    conn = sqlite3.connect(db)
    entity = _get_entity(conn, "@badplace")
    conn.close()

    assert entity is None


def test_no_handles_in_text(db):
    from src.ingestion.handle_extractor import HandleExtractor

    extractor = HandleExtractor(db_path=db, max_depth=2, blocklist=[], logger=_make_logger())
    extractor.process("No social handles here, just plain text.", source_handle="@seed", source_depth=0)

    conn = sqlite3.connect(db)
    count = conn.execute("SELECT COUNT(*) FROM candidate_entities").fetchone()[0]
    conn.close()

    assert count == 0
