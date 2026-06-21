"""Unit tests for IngestionService."""

from __future__ import annotations

import io
import json
import sqlite3
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import yaml

from src.config import AppConfig, LocationConfig, ScrapingConfig, VenueDiscoveryConfig
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
):
    from src.models.event_candidate import EventCandidate

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
        discovered_at=FIXED_NOW,
    )


def _mock_social_source(candidates):
    from src.ingestion.source import IngestionSource

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
    from src.ingestion.ingestion_service import IngestionService

    svc = IngestionService(
        config=_make_config(),
        db_path=db,
        seeds_path=seeds_yaml,
        social_sources=[],
        movie_sources=[],
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
    from src.ingestion.ingestion_service import IngestionService

    svc = IngestionService(
        config=_make_config(),
        db_path=db,
        seeds_path=seeds_yaml,
        social_sources=[],
        movie_sources=[],
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
    from src.ingestion.ingestion_service import IngestionService

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
        social_sources=[],
        movie_sources=[],
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
    from src.ingestion.ingestion_service import IngestionService

    recent = _make_candidate(days_ago=10)
    svc = IngestionService(
        config=_make_config(lookback_days=30),
        db_path=db,
        seeds_path=seeds_yaml,
        social_sources=[_mock_social_source([recent])],
        movie_sources=[],
        logger=_make_logger(),
    )
    result = svc.run(get_now=lambda: FIXED_NOW)

    conn = sqlite3.connect(db)
    rows = _get_persisted_candidates(conn)
    conn.close()

    assert len(rows) == 1


def test_old_post_discarded(db, seeds_yaml):
    from src.ingestion.ingestion_service import IngestionService

    old = _make_candidate(days_ago=40)
    svc = IngestionService(
        config=_make_config(lookback_days=30),
        db_path=db,
        seeds_path=seeds_yaml,
        social_sources=[_mock_social_source([old])],
        movie_sources=[],
        logger=_make_logger(),
    )
    svc.run(get_now=lambda: FIXED_NOW)

    conn = sqlite3.connect(db)
    rows = _get_persisted_candidates(conn)
    conn.close()

    assert len(rows) == 0


def test_none_published_at_bypasses_lookback(db, seeds_yaml):
    """Movie schedules (raw_published_at=None) always pass the lookback filter."""
    from src.ingestion.ingestion_service import IngestionService

    movie = _make_candidate(source_type="cinema_veezi", raw_published_at=None)
    svc = IngestionService(
        config=_make_config(lookback_days=30),
        db_path=db,
        seeds_path=seeds_yaml,
        social_sources=[],
        movie_sources=[_mock_social_source([movie])],
        logger=_make_logger(),
    )
    svc.run(get_now=lambda: FIXED_NOW)

    conn = sqlite3.connect(db)
    rows = _get_persisted_candidates(conn)
    conn.close()

    assert len(rows) == 1


def test_lookback_reads_from_config(db, seeds_yaml):
    """A post 20 days old: passes with lookback=30, discarded with lookback=10."""
    from src.ingestion.ingestion_service import IngestionService

    ec = _make_candidate(days_ago=20)

    svc_30 = IngestionService(
        config=_make_config(lookback_days=30),
        db_path=db,
        seeds_path=seeds_yaml,
        social_sources=[_mock_social_source([ec])],
        movie_sources=[],
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
        social_sources=[_mock_social_source([ec2])],
        movie_sources=[],
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
    from src.ingestion.ingestion_service import IngestionService
    from src.models.event_candidate import EventCandidate

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
        social_sources=[_mock_social_source([malformed])],
        movie_sources=[],
        logger=_make_logger(),
    )
    result = svc.run(get_now=lambda: FIXED_NOW)

    conn = sqlite3.connect(db)
    rows = _get_persisted_candidates(conn)
    conn.close()

    assert len(rows) == 0
    assert result.discarded >= 1


def test_record_missing_only_title_retained(db, seeds_yaml):
    from src.ingestion.ingestion_service import IngestionService
    from src.models.event_candidate import EventCandidate

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
        social_sources=[_mock_social_source([ec])],
        movie_sources=[],
        logger=_make_logger(),
    )
    svc.run(get_now=lambda: FIXED_NOW)

    conn = sqlite3.connect(db)
    rows = _get_persisted_candidates(conn)
    conn.close()

    assert len(rows) == 1


def test_one_malformed_does_not_stop_ingestion(db, seeds_yaml):
    from src.ingestion.ingestion_service import IngestionService
    from src.models.event_candidate import EventCandidate

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
        social_sources=[_mock_social_source([malformed, good1, good2])],
        movie_sources=[],
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
    from src.ingestion.ingestion_service import IngestionService

    ec = _make_candidate(title="Jazz Night", days_ago=5)
    svc = IngestionService(
        config=_make_config(),
        db_path=db,
        seeds_path=seeds_yaml,
        social_sources=[_mock_social_source([ec])],
        movie_sources=[],
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
    from src.ingestion.ingestion_service import IngestionService

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
        social_sources=[],
        movie_sources=[],
        logger=_make_logger(),
    )
    svc.run(get_now=lambda: FIXED_NOW)

    conn = sqlite3.connect(db)
    entities = _get_candidate_entities(conn)
    conn.close()

    assert entities["@promoteme"]["state"] == "active"


def test_handle_not_promoted_without_seed_source(db, seeds_yaml):
    from src.ingestion.ingestion_service import IngestionService

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
        social_sources=[],
        movie_sources=[],
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
    from src.ingestion.ingestion_service import IngestionService
    from src.ingestion.source import IngestionSource

    failing = MagicMock(spec=IngestionSource)
    failing.fetch.side_effect = RuntimeError("network error")

    good = _mock_social_source([_make_candidate(days_ago=5)])

    svc = IngestionService(
        config=_make_config(),
        db_path=db,
        seeds_path=seeds_yaml,
        social_sources=[failing, good],
        movie_sources=[],
        logger=_make_logger(),
    )
    # Must not raise; failover handles it
    result = svc.run(get_now=lambda: FIXED_NOW)

    conn = sqlite3.connect(db)
    rows = _get_persisted_candidates(conn)
    conn.close()

    assert len(rows) == 1
