"""Unit tests for event candidate reload."""

from __future__ import annotations

import io
import sqlite3
from datetime import datetime, timedelta, timezone

import yaml

import pytest

from src.config import AppConfig, LocationConfig, ScrapingConfig, VenueDiscoveryConfig
from src.ingestion.ingestion_service import IngestionService
from src.ingestion.source import IngestionSource
from src.models.event_candidate import EventCandidate
from src.models.tag import Tag
from src.storage.sqlite.candidates import SqliteCandidateRepository
from src.storage.db import init_db

_NOW = datetime(2026, 8, 11, 12, 0, tzinfo=timezone.utc)


def _db(tmp_path):
    path = tmp_path / "candidates.db"
    init_db(path)
    return path


def _listing_candidate(**kwargs) -> EventCandidate:
    defaults = dict(
        id="c1",
        source="northshorenightout_listing",
        source_type="northshorenightout",
        discovered_at=_NOW,
        title="Trivia",
        start_time=_NOW,
    )
    defaults.update(kwargs)
    return EventCandidate(**defaults)


def _reload(db) -> EventCandidate:
    loaded = SqliteCandidateRepository(db).for_window(
        discovered_since=_NOW - timedelta(days=1),
        starting_after=_NOW - timedelta(days=1),
    )
    assert len(loaded) == 1
    return loaded[0]
from src.utils.logging import get_logger

NOW = datetime(2025, 6, 15, 12, 0, 0, tzinfo=timezone.utc)
LOOKBACK = NOW - timedelta(days=30)


@pytest.fixture
def db(tmp_path):
    path = tmp_path / "test.db"
    init_db(db_path=path)
    return path


def _insert(db_path, candidate: EventCandidate) -> None:
    conn = sqlite3.connect(db_path)
    conn.execute(
        """INSERT OR REPLACE INTO event_candidates
           (id, source, source_type, url, image_url, raw_published_at,
            title, description, venue, location, start_time, end_time, discovered_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            candidate.id,
            candidate.source,
            candidate.source_type,
            candidate.url,
            candidate.image_url,
            candidate.raw_published_at.isoformat() if candidate.raw_published_at else None,
            candidate.title,
            candidate.description,
            candidate.venue,
            candidate.location,
            candidate.start_time.isoformat() if candidate.start_time else None,
            candidate.end_time.isoformat() if candidate.end_time else None,
            candidate.discovered_at.isoformat(),
        ),
    )
    conn.commit()
    conn.close()


def _candidate(candidate_id: str, **overrides) -> EventCandidate:
    fields = {
        "id": candidate_id,
        "source": "@venue",
        "source_type": "apify",
        "discovered_at": NOW,
    }
    fields.update(overrides)
    return EventCandidate(**fields)


def _load(db_path):
    return SqliteCandidateRepository(db_path).for_window(
        discovered_since=LOOKBACK, starting_after=NOW
    )


def test_empty_table_returns_empty_list(db):
    assert _load(db) == []


def test_every_field_round_trips(db):
    original = _candidate(
        "c1",
        url="https://example.com/post",
        image_url="https://cdn.example.com/i.jpg",
        raw_published_at=NOW - timedelta(days=2),
        title="Jazz Night",
        description="Live jazz from 8pm",
        venue="The Vault",
        location="123 Main St",
        start_time=NOW + timedelta(days=1),
        end_time=NOW + timedelta(days=1, hours=3),
    )
    _insert(db, original)

    assert _load(db) == [original]


def test_sparse_candidate_round_trips(db):
    original = _candidate("c1", description="Something is happening")
    _insert(db, original)

    assert _load(db) == [original]


def test_datetimes_come_back_timezone_aware(db):
    _insert(db, _candidate("c1", start_time=NOW + timedelta(days=1)))

    loaded = _load(db)[0]
    assert loaded.discovered_at.tzinfo is not None
    assert loaded.start_time.tzinfo is not None


def test_recently_discovered_undated_candidate_is_included(db):
    """Social candidates carry no start_time; a forward-only filter drops them all."""
    _insert(db, _candidate("c1", discovered_at=NOW - timedelta(days=3)))

    assert [c.id for c in _load(db)] == ["c1"]


def test_stale_undated_candidate_is_excluded(db):
    _insert(db, _candidate("c1", discovered_at=NOW - timedelta(days=45)))

    assert _load(db) == []


def test_stale_candidate_still_upcoming_is_included(db):
    """The union's whole point: a calendar event found long ago has not happened yet."""
    _insert(
        db,
        _candidate(
            "c1",
            discovered_at=NOW - timedelta(days=45),
            start_time=NOW + timedelta(days=10),
        ),
    )

    assert [c.id for c in _load(db)] == ["c1"]


def test_stale_candidate_already_past_is_excluded(db):
    _insert(
        db,
        _candidate(
            "c1",
            discovered_at=NOW - timedelta(days=45),
            start_time=NOW - timedelta(days=40),
        ),
    )

    assert _load(db) == []


def test_recently_discovered_past_event_is_included(db):
    """The discovered arm alone qualifies it; the two arms are a union, not a filter pair."""
    _insert(
        db,
        _candidate(
            "c1",
            discovered_at=NOW - timedelta(days=1),
            start_time=NOW - timedelta(hours=6),
        ),
    )

    assert [c.id for c in _load(db)] == ["c1"]


def test_candidate_starting_exactly_now_is_included(db):
    _insert(db, _candidate("c1", discovered_at=NOW - timedelta(days=45), start_time=NOW))

    assert [c.id for c in _load(db)] == ["c1"]


def test_results_are_ordered_by_discovery_then_id(db):
    """Dedup picks a merge base partly on ordering, so the read must not vary.

    Insertion order is deliberately the reverse of the expected order: a tie on
    `discovered_at` breaks on id, never on whatever order SQLite scanned in.
    """
    _insert(db, _candidate("c3", discovered_at=NOW - timedelta(days=1)))
    _insert(db, _candidate("c2", discovered_at=NOW - timedelta(days=2)))
    _insert(db, _candidate("c1", discovered_at=NOW - timedelta(days=2)))

    assert [c.id for c in _load(db)] == ["c1", "c2", "c3"]


def test_loads_what_the_ingestion_service_wrote(db, tmp_path):
    """Reader and writer must agree on the columns, not merely on the schema."""
    seeds = tmp_path / "seeds.yaml"
    seeds.write_text(yaml.dump({"handles": ["@seedvenue"], "venues": []}))

    written = _candidate(
        "c1",
        title="Jazz Night",
        description="Live jazz from 8pm",
        venue="The Vault",
        start_time=NOW + timedelta(days=1),
        raw_published_at=NOW - timedelta(days=1),
    )

    class _StubSource(IngestionSource):
        def fetch(self):
            return [written]

    IngestionService(
        config=AppConfig(
            location=LocationConfig(
                latitude=42.52,
                longitude=-70.89,
                postal_code="01970",
                search_radius_miles=10,
                timezone="America/New_York",
            ),
            scraping=ScrapingConfig(
                lookback_days=30,
                max_discovery_depth=2,
                candidate_promotion_threshold=3,
            ),
            venue_discovery=VenueDiscoveryConfig(),
        ),
        db_path=db,
        seeds_path=seeds,
        failover_sources=[_StubSource()],
        independent_sources=[],
        logger=get_logger("test_candidates", stream=io.StringIO()),
    ).run(get_now=lambda: NOW)

    assert _load(db) == [written]
