"""Unit tests for IngestionService."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock
import io
import json
import sqlite3
import uuid

import pytest
import yaml

from src.config import AppConfig, LocationConfig, ScrapingConfig, VenueDiscoveryConfig
from src.ingestion.ingestion_service import IngestionService
from src.ingestion.source import IngestionSource
from src.models.event_candidate import EventCandidate
from src.storage.db import init_db
from src.utils.logging import get_logger


FIXED_NOW = datetime(2025, 6, 15, 12, 0, 0, tzinfo=timezone.utc)


def _make_config(lookback_days: int = 30, promotion_threshold: int = 3) -> AppConfig:
    return AppConfig(
        location=LocationConfig(
            latitude=42.52,
            longitude=-70.89,
            postal_code="01970",
            search_radius_miles=10,
            timezone="America/New_York",
        ),
        scraping=ScrapingConfig(
            lookback_days=lookback_days,
            max_discovery_depth=2,
            candidate_promotion_threshold=promotion_threshold,
        ),
        venue_discovery=VenueDiscoveryConfig(),
        ollama_host="http://localhost:11434",
    )


def _make_logger():
    return get_logger("test_svc", stream=io.StringIO())


@pytest.fixture
def db(tmp_path):
    path = tmp_path / "test.db"
    init_db(db_path=path)
    return path


@pytest.fixture
def seeds_yaml(tmp_path):
    path = tmp_path / "seeds.yaml"
    path.write_text(yaml.dump({"handles": ["@seedvenue"], "venues": []}))
    return path


def _make_candidate(
    title="Test Event",
    description="A great event",
    source="@seedvenue",
    source_type="apify",
    raw_published_at: datetime | None = None,
    days_ago: int | None = None,
    start_time: datetime | None = None,
):

    pub = None
    if days_ago is not None:
        pub = FIXED_NOW - timedelta(days=days_ago)
    elif raw_published_at is not None:
        pub = raw_published_at

    return EventCandidate(
        id=str(uuid.uuid4()),
        source=source,
        source_type=source_type,
        title=title,
        description=description,
        raw_published_at=pub,
        start_time=start_time,
        discovered_at=FIXED_NOW,
    )


def _mock_social_source(candidates):

    src = MagicMock(spec=IngestionSource)
    src.fetch.return_value = candidates
    return src


def _get_persisted_candidates(conn):
    return conn.execute("SELECT id, title FROM event_candidates").fetchall()


def _get_candidate_entities(conn):
    return {
        row[0]: {"state": row[1], "depth": row[2]}
        for row in conn.execute(
            "SELECT handle, state, depth FROM candidate_entities"
        ).fetchall()
    }


# ---------------------------------------------------------------------------
# Seed loading
# ---------------------------------------------------------------------------


def test_seed_handles_loaded_as_active(db, seeds_yaml, tmp_path):

    svc = IngestionService(
        config=_make_config(),
        db_path=db,
        seeds_path=seeds_yaml,
        failover_sources=[],
        independent_sources=[],
        logger=_make_logger(),
    )
    svc.run(get_now=lambda: FIXED_NOW)

    conn = sqlite3.connect(db)
    entities = _get_candidate_entities(conn)
    conn.close()

    assert "@seedvenue" in entities
    assert entities["@seedvenue"]["state"] == "active"
    assert entities["@seedvenue"]["depth"] == 0


def test_seed_load_is_idempotent(db, seeds_yaml):

    svc = IngestionService(
        config=_make_config(),
        db_path=db,
        seeds_path=seeds_yaml,
        failover_sources=[],
        independent_sources=[],
        logger=_make_logger(),
    )
    svc.run(get_now=lambda: FIXED_NOW)
    svc.run(get_now=lambda: FIXED_NOW)

    conn = sqlite3.connect(db)
    count = conn.execute(
        "SELECT COUNT(*) FROM candidate_entities WHERE handle = '@seedvenue'"
    ).fetchone()[0]
    conn.close()

    assert count == 1


def test_probationary_handle_in_seeds_promoted_to_active(db, seeds_yaml):
    """Handle already in candidate_entities as probationary gets promoted to active if in seeds."""

    conn = sqlite3.connect(db)
    conn.execute(
        """INSERT INTO candidate_entities
           (id, handle, state, depth, mention_count, mention_sources, created_at, updated_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            str(uuid.uuid4()),
            "@seedvenue",
            "probationary",
            1,
            0,
            json.dumps([]),
            FIXED_NOW.isoformat(),
            FIXED_NOW.isoformat(),
        ),
    )
    conn.commit()
    conn.close()

    svc = IngestionService(
        config=_make_config(),
        db_path=db,
        seeds_path=seeds_yaml,
        failover_sources=[],
        independent_sources=[],
        logger=_make_logger(),
    )
    svc.run(get_now=lambda: FIXED_NOW)

    conn = sqlite3.connect(db)
    entities = _get_candidate_entities(conn)
    conn.close()

    assert entities["@seedvenue"]["state"] == "active"


# ---------------------------------------------------------------------------
# Lookback window filtering
# ---------------------------------------------------------------------------


def test_recent_post_retained(db, seeds_yaml):

    recent = _make_candidate(days_ago=10)
    svc = IngestionService(
        config=_make_config(lookback_days=30),
        db_path=db,
        seeds_path=seeds_yaml,
        failover_sources=[_mock_social_source([recent])],
        independent_sources=[],
        logger=_make_logger(),
    )
    result = svc.run(get_now=lambda: FIXED_NOW)

    conn = sqlite3.connect(db)
    rows = _get_persisted_candidates(conn)
    conn.close()

    assert len(rows) == 1


def test_old_post_discarded(db, seeds_yaml):

    old = _make_candidate(days_ago=40)
    svc = IngestionService(
        config=_make_config(lookback_days=30),
        db_path=db,
        seeds_path=seeds_yaml,
        failover_sources=[_mock_social_source([old])],
        independent_sources=[],
        logger=_make_logger(),
    )
    svc.run(get_now=lambda: FIXED_NOW)

    conn = sqlite3.connect(db)
    rows = _get_persisted_candidates(conn)
    conn.close()

    assert len(rows) == 0


def _run_with(candidate, db, seeds_yaml, lookback_days=30):
    """Run one ingestion pass over a single candidate, returning persisted rows."""
    svc = IngestionService(
        config=_make_config(lookback_days=lookback_days),
        db_path=db,
        seeds_path=seeds_yaml,
        failover_sources=[_mock_social_source([candidate])],
        independent_sources=[],
        logger=_make_logger(),
    )
    svc.run(get_now=lambda: FIXED_NOW)

    conn = sqlite3.connect(db)
    rows = _get_persisted_candidates(conn)
    conn.close()
    return rows


def test_old_candidate_with_future_start_time_retained(db, seeds_yaml):
    """An event that has not happened yet is never stale, however old its listing.

    Forward-looking sources (public calendars) carry announcement dates that may
    long predate the lookback window while the event itself is still upcoming.
    """
    upcoming = _make_candidate(
        days_ago=400,
        start_time=FIXED_NOW + timedelta(days=5),
    )

    assert len(_run_with(upcoming, db, seeds_yaml)) == 1


def test_old_candidate_with_past_start_time_discarded(db, seeds_yaml):
    """An old listing for an event that has already happened is still discarded."""
    finished = _make_candidate(
        days_ago=40,
        start_time=FIXED_NOW - timedelta(days=2),
    )

    assert len(_run_with(finished, db, seeds_yaml)) == 0


def test_old_candidate_starting_now_discarded(db, seeds_yaml):
    """The boundary is strict: starting exactly now is not 'yet to happen'."""
    starting_now = _make_candidate(days_ago=40, start_time=FIXED_NOW)

    assert len(_run_with(starting_now, db, seeds_yaml)) == 0


def test_old_candidate_without_start_time_discarded(db, seeds_yaml):
    """Social posts carry no start_time at ingestion, so the lookback still governs."""
    old_post = _make_candidate(days_ago=40, start_time=None)

    assert len(_run_with(old_post, db, seeds_yaml)) == 0


def test_none_published_at_bypasses_lookback(db, seeds_yaml):
    """Movie schedules (raw_published_at=None) always pass the lookback filter."""

    movie = _make_candidate(source_type="cinema_veezi", raw_published_at=None)
    svc = IngestionService(
        config=_make_config(lookback_days=30),
        db_path=db,
        seeds_path=seeds_yaml,
        failover_sources=[],
        independent_sources=[_mock_social_source([movie])],
        logger=_make_logger(),
    )
    svc.run(get_now=lambda: FIXED_NOW)

    conn = sqlite3.connect(db)
    rows = _get_persisted_candidates(conn)
    conn.close()

    assert len(rows) == 1


def test_lookback_reads_from_config(db, seeds_yaml):
    """A post 20 days old: passes with lookback=30, discarded with lookback=10."""

    ec = _make_candidate(days_ago=20)

    svc_30 = IngestionService(
        config=_make_config(lookback_days=30),
        db_path=db,
        seeds_path=seeds_yaml,
        failover_sources=[_mock_social_source([ec])],
        independent_sources=[],
        logger=_make_logger(),
    )
    svc_30.run(get_now=lambda: FIXED_NOW)

    conn = sqlite3.connect(db)
    rows = _get_persisted_candidates(conn)
    conn.close()
    assert len(rows) == 1

    db2_path = db.parent / "test2.db"
    init_db(db_path=db2_path)

    ec2 = _make_candidate(days_ago=20)
    svc_10 = IngestionService(
        config=_make_config(lookback_days=10),
        db_path=db2_path,
        seeds_path=seeds_yaml,
        failover_sources=[_mock_social_source([ec2])],
        independent_sources=[],
        logger=_make_logger(),
    )
    svc_10.run(get_now=lambda: FIXED_NOW)

    conn2 = sqlite3.connect(db2_path)
    rows2 = _get_persisted_candidates(conn2)
    conn2.close()
    assert len(rows2) == 0


# ---------------------------------------------------------------------------
# Malformed record handling
# ---------------------------------------------------------------------------


def test_malformed_record_all_key_fields_absent_discarded(db, seeds_yaml):

    malformed = EventCandidate(
        id=str(uuid.uuid4()),
        source="@src",
        source_type="apify",
        discovered_at=FIXED_NOW,
        # title, description, start_time all None
    )
    svc = IngestionService(
        config=_make_config(),
        db_path=db,
        seeds_path=seeds_yaml,
        failover_sources=[_mock_social_source([malformed])],
        independent_sources=[],
        logger=_make_logger(),
    )
    result = svc.run(get_now=lambda: FIXED_NOW)

    conn = sqlite3.connect(db)
    rows = _get_persisted_candidates(conn)
    conn.close()

    assert len(rows) == 0
    assert result.discarded >= 1


def test_record_missing_only_title_retained(db, seeds_yaml):

    ec = EventCandidate(
        id=str(uuid.uuid4()),
        source="@src",
        source_type="apify",
        description="Some description",
        start_time=FIXED_NOW,
        discovered_at=FIXED_NOW,
    )
    svc = IngestionService(
        config=_make_config(),
        db_path=db,
        seeds_path=seeds_yaml,
        failover_sources=[_mock_social_source([ec])],
        independent_sources=[],
        logger=_make_logger(),
    )
    svc.run(get_now=lambda: FIXED_NOW)

    conn = sqlite3.connect(db)
    rows = _get_persisted_candidates(conn)
    conn.close()

    assert len(rows) == 1


def test_one_malformed_does_not_stop_ingestion(db, seeds_yaml):

    malformed = EventCandidate(
        id=str(uuid.uuid4()),
        source="@src",
        source_type="apify",
        discovered_at=FIXED_NOW,
    )
    good1 = _make_candidate(title="Good Event A", days_ago=5)
    good2 = _make_candidate(title="Good Event B", days_ago=5)

    svc = IngestionService(
        config=_make_config(),
        db_path=db,
        seeds_path=seeds_yaml,
        failover_sources=[_mock_social_source([malformed, good1, good2])],
        independent_sources=[],
        logger=_make_logger(),
    )
    result = svc.run(get_now=lambda: FIXED_NOW)

    conn = sqlite3.connect(db)
    rows = _get_persisted_candidates(conn)
    conn.close()

    assert len(rows) == 2
    assert result.discarded == 1


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def test_event_candidates_persisted_to_db(db, seeds_yaml):

    ec = _make_candidate(title="Jazz Night", days_ago=5)
    svc = IngestionService(
        config=_make_config(),
        db_path=db,
        seeds_path=seeds_yaml,
        failover_sources=[_mock_social_source([ec])],
        independent_sources=[],
        logger=_make_logger(),
    )
    svc.run(get_now=lambda: FIXED_NOW)

    conn = sqlite3.connect(db)
    cursor = conn.execute("PRAGMA table_info(event_candidates)")
    col_names = {row[1] for row in cursor.fetchall()}
    row = conn.execute("SELECT * FROM event_candidates LIMIT 1").fetchone()
    conn.close()

    assert row is not None
    assert "raw_published_at" in col_names


# ---------------------------------------------------------------------------
# Handle promotion
# ---------------------------------------------------------------------------


def test_handle_promoted_when_threshold_met_with_seed_source(db, seeds_yaml):

    # Insert a probationary handle that has been mentioned by a seed source enough times
    conn = sqlite3.connect(db)
    conn.execute(
        """INSERT INTO candidate_entities
           (id, handle, state, depth, mention_count, mention_sources,
            llm_classification, created_at, updated_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            str(uuid.uuid4()),
            "@promoteme",
            "probationary",
            1,
            3,
            json.dumps(["@seedvenue"]),  # seed source
            "venue",
            FIXED_NOW.isoformat(),
            FIXED_NOW.isoformat(),
        ),
    )
    conn.commit()
    conn.close()

    svc = IngestionService(
        config=_make_config(promotion_threshold=3),
        db_path=db,
        seeds_path=seeds_yaml,
        failover_sources=[],
        independent_sources=[],
        logger=_make_logger(),
    )
    svc.run(get_now=lambda: FIXED_NOW)

    conn = sqlite3.connect(db)
    entities = _get_candidate_entities(conn)
    conn.close()

    assert entities["@promoteme"]["state"] == "active"


def test_handle_not_promoted_without_seed_source(db, seeds_yaml):

    conn = sqlite3.connect(db)
    conn.execute(
        """INSERT INTO candidate_entities
           (id, handle, state, depth, mention_count, mention_sources,
            llm_classification, created_at, updated_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            str(uuid.uuid4()),
            "@nopromo",
            "probationary",
            1,
            5,
            json.dumps(["@unknownhandle"]),  # not a seed source
            "venue",
            FIXED_NOW.isoformat(),
            FIXED_NOW.isoformat(),
        ),
    )
    conn.commit()
    conn.close()

    svc = IngestionService(
        config=_make_config(promotion_threshold=3),
        db_path=db,
        seeds_path=seeds_yaml,
        failover_sources=[],
        independent_sources=[],
        logger=_make_logger(),
    )
    svc.run(get_now=lambda: FIXED_NOW)

    conn = sqlite3.connect(db)
    entities = _get_candidate_entities(conn)
    conn.close()

    assert entities["@nopromo"]["state"] == "probationary"


# ---------------------------------------------------------------------------
# Provider failure
# ---------------------------------------------------------------------------


def test_social_source_failure_pipeline_continues(db, seeds_yaml):

    failing = MagicMock(spec=IngestionSource)
    failing.fetch.side_effect = RuntimeError("network error")

    good = _mock_social_source([_make_candidate(days_ago=5)])

    svc = IngestionService(
        config=_make_config(),
        db_path=db,
        seeds_path=seeds_yaml,
        failover_sources=[failing, good],
        independent_sources=[],
        logger=_make_logger(),
    )
    # Must not raise; failover handles it
    result = svc.run(get_now=lambda: FIXED_NOW)

    conn = sqlite3.connect(db)
    rows = _get_persisted_candidates(conn)
    conn.close()

    assert len(rows) == 1


# ---------------------------------------------------------------------------
# Fetch without persisting
# ---------------------------------------------------------------------------


def _svc_with(db, seeds_yaml, candidates):
    return IngestionService(
        config=_make_config(),
        db_path=db,
        seeds_path=seeds_yaml,
        failover_sources=[_mock_social_source(candidates)],
        independent_sources=[],
        logger=_make_logger(),
    )


def test_run_returns_the_candidates_it_accepted(db, seeds_yaml):
    candidate = _make_candidate(title="Jazz Night")
    result = _svc_with(db, seeds_yaml, [candidate]).run(get_now=lambda: FIXED_NOW)

    assert [c.title for c in result.candidates] == ["Jazz Night"]


def test_returned_candidates_exclude_discarded_ones(db, seeds_yaml):
    good = _make_candidate(title="Jazz Night")
    old = _make_candidate(title="Ancient", days_ago=400)
    result = _svc_with(db, seeds_yaml, [good, old]).run(get_now=lambda: FIXED_NOW)

    assert [c.title for c in result.candidates] == ["Jazz Night"]
    assert result.accepted == 1
    assert result.discarded == 1


def test_persist_false_writes_no_candidates(db, seeds_yaml):
    """--dry-run must leave the database exactly as it found it."""
    _svc_with(db, seeds_yaml, [_make_candidate()]).run(
        get_now=lambda: FIXED_NOW, persist=False
    )

    conn = sqlite3.connect(db)
    rows = _get_persisted_candidates(conn)
    conn.close()
    assert rows == []


def test_persist_false_writes_no_handles(db, seeds_yaml):
    """Handle discovery writes too, and a dry run must not seed future runs."""
    candidate = _make_candidate(description="Great set by @newvenue last night")
    _svc_with(db, seeds_yaml, [candidate]).run(get_now=lambda: FIXED_NOW, persist=False)

    conn = sqlite3.connect(db)
    entities = _get_candidate_entities(conn)
    conn.close()
    assert entities == {}


def test_persist_false_still_returns_what_it_fetched(db, seeds_yaml):
    """The point of a dry run is proving the providers work."""
    result = _svc_with(db, seeds_yaml, [_make_candidate(title="Jazz Night")]).run(
        get_now=lambda: FIXED_NOW, persist=False
    )

    assert [c.title for c in result.candidates] == ["Jazz Night"]
    assert result.accepted == 1


def test_persisting_run_still_writes(db, seeds_yaml):
    _svc_with(db, seeds_yaml, [_make_candidate()]).run(get_now=lambda: FIXED_NOW)

    conn = sqlite3.connect(db)
    rows = _get_persisted_candidates(conn)
    conn.close()
    assert len(rows) == 1


def test_a_description_mentioning_a_handle_does_not_deadlock(db, seeds_yaml):
    """Handle discovery opens its own connection while candidates are uncommitted.

    `_persist_candidate` writes on the outer connection, which holds a RESERVED
    lock until the commit after the loop. `HandleExtractor` then opens a second
    connection and writes, so it waits on a lock the same call stack is holding
    and dies at the default five-second timeout.

    It only bites when a description actually mentions an @handle, because the
    extractor returns before connecting when it finds none.
    """
    candidate = _make_candidate(description="Tonight at @thejazzclub, doors at 8")

    svc = IngestionService(
        config=_make_config(),
        db_path=db,
        seeds_path=seeds_yaml,
        failover_sources=[_mock_social_source([candidate])],
        independent_sources=[],
        logger=_make_logger(),
    )

    result = svc.run(get_now=lambda: FIXED_NOW)

    assert result.accepted == 1
    conn = sqlite3.connect(db)
    try:
        handles = {
            row[0]
            for row in conn.execute("SELECT handle FROM candidate_entities").fetchall()
        }
    finally:
        conn.close()
    assert "@thejazzclub" in handles
