"""Unit tests for NormalizationService — orchestration and DB persistence."""

from __future__ import annotations

import io
import sqlite3
import uuid
from datetime import datetime, timezone

import pytest

from src.config import AppConfig, DeduplicationConfig, LocationConfig, ScrapingConfig, VenueDiscoveryConfig
from src.models.event_candidate import EventCandidate
from src.normalization.service import NormalizationService
from src.storage.db import init_db
from src.utils.logging import get_logger


_TZ = "America/New_York"


def _cfg() -> AppConfig:
    return AppConfig(
        location=LocationConfig(42.52, -70.89, "01970", 10.0, _TZ),
        scraping=ScrapingConfig(),
        venue_discovery=VenueDiscoveryConfig(),
        deduplication=DeduplicationConfig(),
    )


def _now() -> datetime:
    return datetime(2025, 6, 15, 12, 0, 0, tzinfo=timezone.utc)


def _candidate(**kwargs) -> EventCandidate:
    defaults = dict(
        id=str(uuid.uuid4()),
        source="@test",
        source_type="apify",
        discovered_at=_now(),
        title="Jazz Night",
        start_time=datetime(2025, 6, 15, 20, 0, 0, tzinfo=timezone.utc),
    )
    defaults.update(kwargs)
    return EventCandidate(**defaults)


def _make_service(tmp_path, cfg=None):
    db_path = tmp_path / "test.db"
    init_db(db_path)
    logger = get_logger("test", stream=io.StringIO())
    return NormalizationService(config=cfg or _cfg(), db_path=db_path, logger=logger), db_path


def test_valid_candidates_persisted_to_events_table(tmp_path):
    svc, db_path = _make_service(tmp_path)
    candidates = [_candidate(title="Jazz Night"), _candidate(title="Trivia Tuesday")]
    result = svc.run(candidates, get_now=_now)
    assert result.persisted == 2
    conn = sqlite3.connect(db_path)
    rows = conn.execute("SELECT title FROM events").fetchall()
    conn.close()
    titles = {r[0] for r in rows}
    assert "Jazz Night" in titles
    assert "Trivia Tuesday" in titles


def test_malformed_candidate_not_persisted(tmp_path):
    svc, db_path = _make_service(tmp_path)
    bad = _candidate(title=None, start_time=None)
    result = svc.run([bad], get_now=_now)
    assert result.persisted == 0
    assert result.discarded == 1
    conn = sqlite3.connect(db_path)
    count = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
    conn.close()
    assert count == 0


def test_duplicate_candidates_merged_to_one_event(tmp_path):
    svc, db_path = _make_service(tmp_path)
    a = _candidate(title="Jazz Night")
    b = _candidate(title="Jazz Night")
    result = svc.run([a, b], get_now=_now)
    assert result.persisted == 1
    conn = sqlite3.connect(db_path)
    count = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
    conn.close()
    assert count == 1


def test_source_event_candidates_stored_as_json(tmp_path):
    import json
    svc, db_path = _make_service(tmp_path)
    cand = _candidate(id="cand-abc")
    svc.run([cand], get_now=_now)
    conn = sqlite3.connect(db_path)
    row = conn.execute("SELECT source_event_candidates FROM events").fetchone()
    conn.close()
    ids = json.loads(row[0])
    assert "cand-abc" in ids


def test_result_counts_correct_mixed_batch(tmp_path):
    svc, db_path = _make_service(tmp_path)
    good = _candidate(title="Good Event")
    bad = _candidate(title=None, start_time=None)
    result = svc.run([good, bad], get_now=_now)
    assert result.persisted == 1
    assert result.discarded == 1


def test_empty_candidates_returns_zero_counts(tmp_path):
    svc, db_path = _make_service(tmp_path)
    result = svc.run([], get_now=_now)
    assert result.persisted == 0
    assert result.discarded == 0


def test_discard_logged_with_source_and_reason(tmp_path):
    """Discarded candidates log both the source handle and the reason."""
    import json
    log_stream = io.StringIO()
    db_path = tmp_path / "test.db"
    init_db(db_path)
    logger = get_logger("test", stream=log_stream)
    svc = NormalizationService(config=_cfg(), db_path=db_path, logger=logger)

    bad = _candidate(title=None, start_time=None, source="@bad_source")
    svc.run([bad], get_now=_now)

    log_stream.seek(0)
    entries = [json.loads(line) for line in log_stream if line.strip()]
    assert any(
        "@bad_source" in e.get("message", "") and "start_time" in e.get("message", "")
        for e in entries
    ), f"Expected discard log with source and reason, got: {entries}"
