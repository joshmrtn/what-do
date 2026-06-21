"""Unit tests for handle disambiguation step (step 3a)."""

from __future__ import annotations

import io
import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.storage.db import init_db
from src.utils.logging import get_logger


def _make_logger():
    return get_logger("test_disambig", stream=io.StringIO())


@pytest.fixture
def db(tmp_path):
    path = tmp_path / "test.db"
    init_db(db_path=path)
    return path


def _insert_candidate(conn, handle, state="probationary", llm_classification=None):
    conn.execute(
        """INSERT INTO candidate_entities
           (id, handle, state, depth, mention_count, mention_sources, llm_classification,
            created_at, updated_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            str(uuid.uuid4()),
            handle,
            state,
            1,
            0,
            json.dumps([]),
            llm_classification,
            datetime.now(timezone.utc).isoformat(),
            datetime.now(timezone.utc).isoformat(),
        ),
    )
    conn.commit()


def _get_state(conn, handle):
    row = conn.execute(
        "SELECT state, llm_classification FROM candidate_entities WHERE handle = ?",
        (handle,),
    ).fetchone()
    return {"state": row[0], "llm_classification": row[1]} if row else None


def _make_provider(classification: str):
    from src.ingestion.disambiguation import DisambiguationProvider

    p = MagicMock(spec=DisambiguationProvider)
    p.classify.return_value = classification
    return p


# ---------------------------------------------------------------------------
# DisambiguationProvider ABC
# ---------------------------------------------------------------------------


def test_disambiguation_provider_is_abstract():
    from src.ingestion.disambiguation import DisambiguationProvider

    with pytest.raises(TypeError):
        DisambiguationProvider()  # type: ignore[abstract]


def test_disambiguation_provider_requires_classify():
    from src.ingestion.disambiguation import DisambiguationProvider

    class Incomplete(DisambiguationProvider):
        pass

    with pytest.raises(TypeError):
        Incomplete()


# ---------------------------------------------------------------------------
# Step 3a orchestration
# ---------------------------------------------------------------------------


def test_person_handle_gets_discarded(db):
    from src.ingestion.disambiguation import DisambiguationStep

    conn = sqlite3.connect(db)
    _insert_candidate(conn, "@johndoe")

    step = DisambiguationStep(
        db_path=db,
        provider=_make_provider("person"),
        logger=_make_logger(),
    )
    step.run()

    entity = _get_state(conn, "@johndoe")
    conn.close()

    assert entity["state"] == "discarded"
    assert entity["llm_classification"] == "person"


def test_venue_handle_stays_probationary(db):
    from src.ingestion.disambiguation import DisambiguationStep

    conn = sqlite3.connect(db)
    _insert_candidate(conn, "@jazzclub")

    step = DisambiguationStep(
        db_path=db,
        provider=_make_provider("venue"),
        logger=_make_logger(),
    )
    step.run()

    entity = _get_state(conn, "@jazzclub")
    conn.close()

    assert entity["state"] == "probationary"
    assert entity["llm_classification"] == "venue"


def test_already_classified_handle_skipped(db):
    from src.ingestion.disambiguation import DisambiguationStep

    conn = sqlite3.connect(db)
    _insert_candidate(conn, "@alreadydone", state="active", llm_classification="venue")

    provider = _make_provider("person")
    step = DisambiguationStep(db_path=db, provider=provider, logger=_make_logger())
    step.run()

    # Provider should NOT have been called
    provider.classify.assert_not_called()
    entity = _get_state(conn, "@alreadydone")
    conn.close()
    assert entity["state"] == "active"  # unchanged


def test_discarded_handle_not_reclassified(db):
    from src.ingestion.disambiguation import DisambiguationStep

    conn = sqlite3.connect(db)
    _insert_candidate(conn, "@oldspam", state="discarded", llm_classification="person")

    provider = _make_provider("venue")
    step = DisambiguationStep(db_path=db, provider=provider, logger=_make_logger())
    step.run()

    provider.classify.assert_not_called()
    entity = _get_state(conn, "@oldspam")
    conn.close()
    assert entity["state"] == "discarded"


def test_provider_failure_leaves_handle_probationary(db):
    from src.ingestion.disambiguation import DisambiguationStep, DisambiguationProvider

    conn = sqlite3.connect(db)
    _insert_candidate(conn, "@flaky")

    provider = MagicMock(spec=DisambiguationProvider)
    provider.classify.side_effect = RuntimeError("LLM unavailable")

    step = DisambiguationStep(db_path=db, provider=provider, logger=_make_logger())
    step.run()  # must not raise

    entity = _get_state(conn, "@flaky")
    conn.close()
    assert entity["state"] == "probationary"


def test_multiple_handles_processed(db):
    from src.ingestion.disambiguation import DisambiguationStep, DisambiguationProvider

    conn = sqlite3.connect(db)
    _insert_candidate(conn, "@venue1")
    _insert_candidate(conn, "@person1")

    provider = MagicMock(spec=DisambiguationProvider)
    provider.classify.side_effect = lambda handle, context: (
        "venue" if handle == "@venue1" else "person"
    )

    step = DisambiguationStep(db_path=db, provider=provider, logger=_make_logger())
    step.run()

    assert _get_state(conn, "@venue1")["state"] == "probationary"
    assert _get_state(conn, "@person1")["state"] == "discarded"
    conn.close()
