"""The candidate writer and reader agree on the columns, not just the schema.

What used to live here was a second, SQLite-only copy of the candidate
repository contract; those cases now run against both implementations in
`test_candidate_repository.py`, which is where a contract belongs.

This one stays because it is not a contract test. `IngestionService` writes
through the `write_candidates` escape hatch — it batches candidate writes
into a transaction it already holds — so the writer and the repository's
reader are separate code paths that must be checked against each other. A
field with no column throws nothing: the reader returns the default and
every test passes. `EventCandidate.timing` was lost that way for weeks.
"""

from __future__ import annotations

import io
import sqlite3
from datetime import datetime, timedelta, timezone

import yaml

import pytest

from src.storage.sqlite.entities import SqliteEntityRepository
from src.config import AppConfig, LocationConfig, ScrapingConfig, VenueDiscoveryConfig
from src.ingestion.ingestion_service import IngestionService
from src.ingestion.source import IngestionSource
from src.models.event_candidate import EventCandidate
from src.storage.sqlite.candidates import SqliteCandidateRepository
from src.storage.sqlite.connection import init_db

_NOW = datetime(2026, 8, 11, 12, 0, tzinfo=timezone.utc)


def _db(tmp_path):
    path = tmp_path / "candidates.db"
    init_db(path)
    return path


from src.utils.logging import get_logger

NOW = datetime(2025, 6, 15, 12, 0, 0, tzinfo=timezone.utc)
LOOKBACK = NOW - timedelta(days=30)


@pytest.fixture
def db(tmp_path):
    path = tmp_path / "test.db"
    init_db(db_path=path)
    return path


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
        seen_since=LOOKBACK, starting_after=NOW
    )


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
        entities=SqliteEntityRepository(db),
        seeds_path=seeds,
        failover_sources=[_StubSource()],
        independent_sources=[],
        logger=get_logger("test_candidates", stream=io.StringIO()),
    ).run(get_now=lambda: NOW)

    assert _load(db) == [written]
