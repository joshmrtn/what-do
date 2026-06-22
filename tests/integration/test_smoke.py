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


def test_config_smoke(sample_config):
    """Config loads and exposes typed location data."""
    from src.config import load_config

    cfg = load_config(config_path=sample_config)
    assert isinstance(cfg.location.latitude, float)
    assert cfg.location.latitude == 42.52


def test_db_and_logger_smoke(sample_config, tmp_path):
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


def test_venue_discovery_smoke(sample_config, tmp_path: Path) -> None:
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


def test_ingestion_smoke(tmp_path: Path) -> None:
    """3 valid events + 1 malformed; failover path works when primary adapter raises."""
    import io
    import uuid
    from datetime import datetime, timedelta, timezone
    from unittest.mock import MagicMock

    import yaml

    from src.config import AppConfig, LocationConfig, ScrapingConfig, VenueDiscoveryConfig
    from src.ingestion.ingestion_service import IngestionService
    from src.ingestion.source import IngestionSource
    from src.models.event_candidate import EventCandidate
    from src.storage.db import init_db
    from src.utils.logging import get_logger

    now = datetime(2025, 6, 15, 12, 0, 0, tzinfo=timezone.utc)
    recent = now - timedelta(days=5)

    def _ec(title, description="desc", days_ago=5):
        return EventCandidate(
            id=str(uuid.uuid4()),
            source="@smoke_seed",
            source_type="apify",
            title=title,
            description=description,
            raw_published_at=now - timedelta(days=days_ago),
            discovered_at=now,
        )

    malformed = EventCandidate(
        id=str(uuid.uuid4()),
        source="@smoke_seed",
        source_type="apify",
        discovered_at=now,
    )

    good_source = MagicMock(spec=IngestionSource)
    good_source.fetch.return_value = [_ec("Event A"), _ec("Event B"), _ec("Event C"), malformed]

    seeds_path = tmp_path / "seeds.yaml"
    seeds_path.write_text(yaml.dump({"handles": ["@smoke_seed"], "venues": []}))

    db_path = tmp_path / "smoke3.db"
    init_db(db_path)

    cfg = AppConfig(
        location=LocationConfig(42.52, -70.89, "01970", 10.0, "America/New_York"),
        scraping=ScrapingConfig(lookback_days=30),
        venue_discovery=VenueDiscoveryConfig(),
        ollama_host="http://localhost:11434",
    )

    svc = IngestionService(
        config=cfg,
        db_path=db_path,
        seeds_path=seeds_path,
        social_sources=[good_source],
        movie_sources=[],
        logger=get_logger("smoke3", stream=io.StringIO()),
    )
    result = svc.run(get_now=lambda: now)

    assert result.persisted == 3, f"expected 3 persisted, got {result.persisted}"
    assert result.discarded == 1, f"expected 1 discarded, got {result.discarded}"

    # Failover: primary fails, secondary succeeds
    failing = MagicMock(spec=IngestionSource)
    failing.fetch.side_effect = RuntimeError("provider down")
    fallback = MagicMock(spec=IngestionSource)
    fallback.fetch.return_value = [_ec("Fallback Event")]

    db2 = tmp_path / "smoke3b.db"
    init_db(db2)
    svc2 = IngestionService(
        config=cfg,
        db_path=db2,
        seeds_path=seeds_path,
        social_sources=[failing, fallback],
        movie_sources=[],
        logger=get_logger("smoke3b", stream=io.StringIO()),
    )
    result2 = svc2.run(get_now=lambda: now)

    assert result2.persisted == 1
    failing.fetch.assert_called_once()
    fallback.fetch.assert_called_once()


def test_normalization_smoke(tmp_path: Path) -> None:
    """2 identical candidates + 1 unique + 1 malformed → 2 events, 1 discard, merged attribution."""
    import io
    import uuid
    from datetime import datetime, timezone

    from src.config import AppConfig, DeduplicationConfig, LocationConfig, ScrapingConfig, VenueDiscoveryConfig
    from src.models.event_candidate import EventCandidate
    from src.normalization.service import NormalizationService
    from src.storage.db import init_db
    from src.utils.logging import get_logger

    now = datetime(2025, 6, 15, 12, 0, 0, tzinfo=timezone.utc)
    event_time = datetime(2025, 6, 15, 20, 0, 0, tzinfo=timezone.utc)

    dup_a = EventCandidate(
        id="dup-a",
        source="@source_one",
        source_type="apify",
        discovered_at=now,
        title="Jazz Night",
        venue="The Vault",
        start_time=event_time,
    )
    dup_b = EventCandidate(
        id="dup-b",
        source="@source_two",
        source_type="apify",
        discovered_at=now,
        title="Jazz Night",
        venue="The Vault",
        start_time=event_time,
    )
    unique = EventCandidate(
        id="unique-1",
        source="@source_one",
        source_type="apify",
        discovered_at=now,
        title="Trivia Tuesday",
        venue="The Anchor",
        start_time=event_time,
    )
    malformed = EventCandidate(
        id="bad-1",
        source="@source_one",
        source_type="apify",
        discovered_at=now,
    )

    db_path = tmp_path / "smoke4.db"
    init_db(db_path)

    cfg = AppConfig(
        location=LocationConfig(42.52, -70.89, "01970", 10.0, "America/New_York"),
        scraping=ScrapingConfig(),
        venue_discovery=VenueDiscoveryConfig(),
        deduplication=DeduplicationConfig(),
    )

    log_stream = io.StringIO()
    svc = NormalizationService(
        config=cfg,
        db_path=db_path,
        logger=get_logger("smoke4", stream=log_stream),
    )
    result = svc.run([dup_a, dup_b, unique, malformed], get_now=lambda: now)

    assert result.persisted == 2, f"expected 2 persisted, got {result.persisted}"
    assert result.discarded == 1, f"expected 1 discarded, got {result.discarded}"

    conn = sqlite3.connect(db_path)
    rows = conn.execute("SELECT title, source_event_candidates FROM events ORDER BY title").fetchall()
    conn.close()

    assert len(rows) == 2
    titles = {r[0] for r in rows}
    assert "Jazz Night" in titles
    assert "Trivia Tuesday" in titles

    jazz_row = next(r for r in rows if r[0] == "Jazz Night")
    attribution = json.loads(jazz_row[1])
    assert set(attribution) == {"dup-a", "dup-b"}, (
        f"merged event should attribute both sources, got {attribution}"
    )

    log_stream.seek(0)
    log_lines = log_stream.read()
    assert "discard" in log_lines.lower() or "missing" in log_lines.lower()


def test_enrichment_smoke(tmp_path: Path) -> None:
    """Enrichment attaches weather/solar to a real event and injects a synthetic activity."""
    import zoneinfo
    from datetime import date, datetime, timezone
    from unittest.mock import MagicMock

    from src.config import (
        AppConfig,
        LocationConfig,
        ScrapingConfig,
        SyntheticActivityRule,
        SyntheticConditions,
        VenueDiscoveryConfig,
    )
    from src.enrichment.astronomical import AstronomicalCalculator
    from src.enrichment.service import EnrichmentService
    from src.enrichment.weather import WeatherProvider
    from src.models.event import Event
    from src.storage.db import init_db

    run_date = date(2025, 6, 21)
    now = datetime(2025, 6, 21, 12, 0, tzinfo=timezone.utc)
    local_tz = zoneinfo.ZoneInfo("America/New_York")

    # Event happening tomorrow (within forecast window)
    tomorrow = datetime(2025, 6, 22, 19, 0, tzinfo=local_tz)
    event = Event(
        event_id="smoke5-evt",
        source_event_candidates=[],
        source_type="instagram",
        created_at=now,
        updated_at=now,
        title="Jazz Night",
        start_time=tomorrow,
    )

    # Mock weather provider returning clear 70°F
    clear_weather = {
        "temperature_f": 70.0,
        "condition": "clear",
        "precipitation_mm": 0.0,
        "wind_speed_mph": 5.0,
    }
    mock_weather = MagicMock(spec=WeatherProvider)
    mock_weather.fetch.return_value = clear_weather

    # Synthetic rule that matches clear weather at ≥60°F
    walk_rule = SyntheticActivityRule(
        name="Evening walk",
        conditions=SyntheticConditions(
            min_temp_f=60.0,
            weather=["clear", "partly_cloudy"],
        ),
        tags=["outdoor", "walking", "low_key"],
        summary="A pleasant evening walk",
    )

    cfg = AppConfig(
        location=LocationConfig(42.52, -70.89, "01970", 10.0, "America/New_York"),
        scraping=ScrapingConfig(),
        venue_discovery=VenueDiscoveryConfig(),
    )

    db_path = tmp_path / "smoke5.db"
    init_db(db_path)

    svc = EnrichmentService(
        weather_provider=mock_weather,
        movie_provider=None,
        astronomical_calculator=AstronomicalCalculator(),
        synthetic_rules=[walk_rule],
        config=cfg,
        db_path=db_path,
        get_now=lambda: now,
    )

    results = svc.enrich([event], run_date)

    # Real event assertions
    assert results[0].weather is not None, "weather should be attached"
    assert results[0].weather["temperature_f"] == 70.0
    assert results[0].astronomical_data is not None, "astronomical_data should be attached"
    assert "sunrise" in results[0].astronomical_data
    assert "sunset" in results[0].astronomical_data
    assert "dawn" in results[0].astronomical_data
    assert "dusk" in results[0].astronomical_data

    # Synthetic event assertions
    synthetic = [e for e in results if e.source_type == "synthetic"]
    assert len(synthetic) == 1, f"expected 1 synthetic event, got {len(synthetic)}"
    syn = synthetic[0]
    assert syn.source_type == "synthetic"
    assert syn.tags == ["outdoor", "walking", "low_key"]
    assert syn.summary == "A pleasant evening walk"
    assert "evening_walk" in syn.event_id
