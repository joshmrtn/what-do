"""Unit tests for NormalizationService — orchestration and discard reporting."""

from __future__ import annotations

from datetime import datetime, timezone
import io
import json
import sqlite3
import uuid

import pytest

from src.config import (
    AppConfig,
    DeduplicationConfig,
    LocationConfig,
    ScrapingConfig,
    VenueDiscoveryConfig,
)
from src.models.event import Event
from src.models.event_candidate import EventCandidate
from src.normalization.service import NormalizationService
from src.storage.sqlite.connection import init_db
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
    return NormalizationService(config=cfg or _cfg(), logger=logger), db_path


def test_valid_candidates_are_returned(tmp_path):
    svc, _ = _make_service(tmp_path)
    candidates = [_candidate(title="Jazz Night"), _candidate(title="Trivia Tuesday")]
    result = svc.run(candidates, get_now=_now)
    assert result.normalized == 2
    titles = {e.title for e in result.events}
    assert "Jazz Night" in titles
    assert "Trivia Tuesday" in titles


def test_events_are_not_persisted(tmp_path):
    """Identity is not settled until reconcile, so the orchestrator owns the save.

    Persisting here would write the normalizer's throwaway uuids, and reconcile
    adopting a stored id afterwards would leave those rows behind as orphans.
    """
    svc, db_path = _make_service(tmp_path)
    svc.run([_candidate(title="Jazz Night")], get_now=_now)

    conn = sqlite3.connect(db_path)
    count = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
    conn.close()
    assert count == 0


def test_malformed_candidate_not_returned(tmp_path):
    svc, _ = _make_service(tmp_path)
    bad = _candidate(title=None, start_time=None)
    result = svc.run([bad], get_now=_now)
    assert result.normalized == 0
    assert result.discarded == 1
    assert result.events == []


def test_duplicate_candidates_merged_to_one_event(tmp_path):
    svc, _ = _make_service(tmp_path)
    a = _candidate(title="Jazz Night")
    b = _candidate(title="Jazz Night")
    result = svc.run([a, b], get_now=_now)
    assert result.normalized == 1
    assert len(result.events) == 1


def test_source_event_candidates_attached_to_the_event(tmp_path):
    svc, _ = _make_service(tmp_path)
    cand = _candidate(id="cand-abc")
    result = svc.run([cand], get_now=_now)
    assert "cand-abc" in result.events[0].source_event_candidates


def test_result_counts_correct_mixed_batch(tmp_path):
    svc, _ = _make_service(tmp_path)
    good = _candidate(title="Good Event")
    bad = _candidate(title=None, start_time=None)
    result = svc.run([good, bad], get_now=_now)
    assert result.normalized == 1
    assert result.discarded == 1


def test_empty_candidates_returns_zero_counts(tmp_path):
    svc, _ = _make_service(tmp_path)
    result = svc.run([], get_now=_now)
    assert result.normalized == 0
    assert result.discarded == 0
    assert result.events == []


def test_discard_logged_with_source_and_reason(tmp_path):
    """Discarded candidates log both the source handle and the reason."""
    log_stream = io.StringIO()
    logger = get_logger("test", stream=log_stream)
    svc = NormalizationService(config=_cfg(), logger=logger)

    bad = _candidate(title=None, start_time=None, source="@bad_source")
    svc.run([bad], get_now=_now)

    log_stream.seek(0)
    entries = [json.loads(line) for line in log_stream if line.strip()]
    assert any(
        "@bad_source" in e.get("message", "") and "start_time" in e.get("message", "")
        for e in entries
    ), f"Expected discard log with source and reason, got: {entries}"
