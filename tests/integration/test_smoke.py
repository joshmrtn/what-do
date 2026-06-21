"""
Smoke tests — verify end-to-end handoffs between components.
Use real local resources (SQLite, config files) but never make external network calls.
One test per phase; they accumulate as phases complete.
"""

import json
import sqlite3
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import yaml


@pytest.fixture
def sample_config(tmp_path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        yaml.dump({
            "location": {
                "latitude": 42.52,
                "longitude": -70.89,
                "postal_code": "01970",
                "search_radius_miles": 10,
            }
        })
    )
    return config_file


def test_phase0_config_smoke(sample_config):
    """Config loads and exposes typed location data."""
    from src.config import load_config

    cfg = load_config(config_path=sample_config)
    assert isinstance(cfg.location.latitude, float)
    assert cfg.location.latitude == 42.52


def test_phase1_db_and_logger_smoke(sample_config, tmp_path):
    """DB initialises and logger writes a structured entry without error."""
    import io
    import json

    from src.storage.db import init_db
    from src.utils.logging import get_logger

    init_db(db_path=tmp_path / "smoke.db")

    stream = io.StringIO()
    log = get_logger("smoke", stream=stream)
    log.info("Phase 1 smoke test", component="smoke", duration_ms=0)

    stream.seek(0)
    entry = json.loads(stream.readline())
    assert entry["message"] == "Phase 1 smoke test"
    assert entry["component"] == "smoke"


def test_phase2_venue_discovery_smoke(sample_config, tmp_path: Path) -> None:
    """Venue discovery persists a seed venue and a provider venue end-to-end."""
    import io

    from src.ingestion.geocoder import GeocoderProvider
    from src.ingestion.venue_discovery import VenueDiscoveryService
    from src.ingestion.venue_source import VenueSource
    from src.models.venue import Venue
    from src.config import load_config
    from src.storage.db import init_db
    from src.utils.logging import get_logger

    # Seed with one handle and one venue
    seeds_path = tmp_path / "seeds.yaml"
    seeds_path.write_text(
        yaml.dump({
            "handles": ["@cinemasalem"],
            "venues": [{"name": "Cinema Salem", "address": "95 Washington St, Salem MA"}],
        })
    )

    blocklist_path = tmp_path / "blocklist.json"
    blocklist_path.write_text("[]")

    db_path = tmp_path / "smoke.db"
    init_db(db_path)

    cfg = load_config(config_path=sample_config)

    # Mock provider returns one nearby venue
    provider = MagicMock(spec=VenueSource)
    provider.fetch_venues.return_value = [
        Venue(
            name="The Vault Lounge",
            address="1 Pickering Wharf",
            latitude=42.520,
            longitude=-70.897,
            category="music_venue",
            social_handles=["@thevaultlounge"],
            discovery_source="mock_provider",
        )
    ]

    # Geocoder resolves the seed venue address
    geocoder = MagicMock(spec=GeocoderProvider)
    geocoder.geocode.return_value = (42.519, -70.896)

    svc = VenueDiscoveryService(
        config=cfg,
        db_path=db_path,
        seeds_path=seeds_path,
        blocklist_path=blocklist_path,
        sources=[provider],
        geocoder=geocoder,
        logger=get_logger("smoke", stream=io.StringIO()),
    )
    svc.run()

    conn = sqlite3.connect(db_path)
    venue_names = [r[0] for r in conn.execute("SELECT name FROM venues").fetchall()]
    handles = [r[0] for r in conn.execute("SELECT handle FROM candidate_entities").fetchall()]
    conn.close()

    assert "Cinema Salem" in venue_names, "seed venue should be persisted"
    assert "The Vault Lounge" in venue_names, "provider venue should be persisted"
    assert "@cinemasalem" in handles, "seed handle should be in candidate_entities"
    assert len(venue_names) == 2
