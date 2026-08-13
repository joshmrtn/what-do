"""
Smoke tests — verify end-to-end handoffs between components.
Use real local resources (SQLite, config files) but never make external network calls.
One test per phase; they accumulate as phases complete.
"""

from datetime import date, datetime, timedelta, timezone
import re
import time
from pathlib import Path
from unittest.mock import MagicMock
import io
import json
import random
import sqlite3
import uuid
import zoneinfo

import pytest
import yaml

from src.storage.sqlite.weather_cache import SqliteWeatherCache
from src.storage.sqlite.entities import SqliteEntityRepository
from src.config import (
    AppConfig,
    DeduplicationConfig,
    LocationConfig,
    ScoringConfig,
    ScrapingConfig,
    SyntheticActivityRule,
    SyntheticConditions,
    VenueDiscoveryConfig,
    load_config,
)
from src.enrichment.astronomical import AstronomicalCalculator
from src.enrichment.comfort import compute_comfort
from src.enrichment.service import EnrichmentService
from src.enrichment.weather import WeatherProvider
from src.ingestion.geocoder import GeocoderProvider
from src.ingestion.ingestion_service import IngestionService
from src.ingestion.source import IngestionSource
from src.ingestion.venue_discovery import VenueDiscoveryService
from src.ingestion.venue_source import VenueSource
from src.models.event import Event
from src.models.event_candidate import EventCandidate
from src.models.tag import Tag
from src.models.venue import Venue
from src.normalization.semantic_dedup import SemanticDeduplicationEngine
from src.normalization.service import NormalizationService
from src.scoring.embedding_stage import EmbeddingStage
from src.scoring.embeddings import OllamaEmbeddingProvider
from src.scoring.preferences import PreferenceRepository
from src.scoring.ranking import RankingEngine
from src.presentation.cli import run
from src.config import FeedConfig
from src.ingestion.calendars.html_source import HtmlListingSource
from src.ingestion.calendars.ics_source import IcsCalendarSource
from src.processing.extraction import ExtractionResult
from src.processing.extraction_stage import ExtractionStage
from src.scheduler import run_batch
from src.scoring.similarity import Reason, SimilarityResult
from src.scoring.similarity_stage import SimilarityStage
from src.storage.sqlite.connection import init_db
from src.storage.sqlite.candidates import SqliteCandidateRepository
from src.storage.sqlite.events import SqliteEventRepository
from src.storage.sqlite.rankings import SqliteRankingRepository
from src.storage.sqlite.runs import SqliteRunRepository
from src.storage.sqlite.scores import SqliteScoreRepository
from src.storage.memory.http_cache import InMemoryHttpCache
from src.storage.events import load_events, save_events
from src.storage.sqlite.rankings import SqliteRankingRepository
from src.storage.sqlite.scores import SqliteScoreRepository
from src.utils.logging import get_logger
from src.utils.ollama_client import OllamaClient
from src.utils.vectors import decode_vector


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

    cfg = load_config(config_path=sample_config)
    assert isinstance(cfg.location.latitude, float)
    assert cfg.location.latitude == 42.52


def test_db_and_logger_smoke(sample_config, tmp_path):
    """DB initialises and logger writes a structured entry without error."""


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
        entities=SqliteEntityRepository(db_path),
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
        entities=SqliteEntityRepository(db_path),
        seeds_path=seeds_path,
        failover_sources=[good_source],
        independent_sources=[],
        logger=get_logger("smoke3", stream=io.StringIO()),
    )
    result = svc.run(get_now=lambda: now)

    assert result.accepted == 3, f"expected 3 accepted, got {result.accepted}"
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
        entities=SqliteEntityRepository(db2),
        seeds_path=seeds_path,
        failover_sources=[failing, fallback],
        independent_sources=[],
        logger=get_logger("smoke3b", stream=io.StringIO()),
    )
    result2 = svc2.run(get_now=lambda: now)

    assert result2.accepted == 1
    failing.fetch.assert_called_once()
    fallback.fetch.assert_called_once()


def test_normalization_smoke(tmp_path: Path) -> None:
    """2 identical candidates + 1 unique + 1 malformed → 2 events, 1 discard, merged attribution."""


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
        logger=get_logger("smoke4", stream=log_stream),
    )
    result = svc.run([dup_a, dup_b, unique, malformed], get_now=lambda: now)

    assert result.normalized == 2, f"expected 2 normalized, got {result.normalized}"
    assert result.discarded == 1, f"expected 1 discarded, got {result.discarded}"

    events = sorted(result.events, key=lambda e: e.title or "")

    assert len(events) == 2
    titles = {e.title for e in events}
    assert "Jazz Night" in titles
    assert "Trivia Tuesday" in titles

    jazz = next(e for e in events if e.title == "Jazz Night")
    assert set(jazz.source_event_candidates) == {"dup-a", "dup-b"}, (
        f"merged event should attribute both sources, got {jazz.source_event_candidates}"
    )

    log_stream.seek(0)
    log_lines = log_stream.read()
    assert "discard" in log_lines.lower() or "missing" in log_lines.lower()


def test_enrichment_smoke(tmp_path: Path) -> None:
    """Enrichment attaches weather/solar to a real event and injects a synthetic activity."""


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

    # Mock weather provider returning a clear 70°F day, hour by hour
    clear_day = {
        "date": "2025-06-22",
        "hours": [
            {
                "hour": hour,
                "temperature_f": 70.0,
                "relative_humidity": 45.0,
                "dew_point_f": 48.0,
                "precipitation_mm": 0.0,
                "wind_speed_mph": 5.0,
                "condition": "clear",
            }
            for hour in range(24)
        ],
    }
    mock_weather = MagicMock(spec=WeatherProvider)
    mock_weather.fetch.return_value = clear_day

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
        weather_cache=SqliteWeatherCache(db_path),
        get_now=lambda: now,
    )

    results = svc.enrich([event], run_date)

    # Real event assertions
    assert results[0].weather is not None, "weather should be attached"
    assert results[0].weather["sampled_hour"] == 19, "sampled at the event's own hour"
    assert results[0].weather["forecast"]["hour"]["temperature_f"] == 70.0
    assert len(results[0].weather["forecast"]["day_series"]) == 24
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
    assert syn.tags == [
        Tag(text="outdoor"),
        Tag(text="walking"),
        Tag(text="low_key"),
    ]
    assert syn.summary == "A pleasant evening walk"
    assert "evening_walk" in syn.event_id


# ---------------------------------------------------------------------------
# Phase 7 — semantic matching engine
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_semantic_matching_smoke(tmp_path: Path) -> None:
    """Embedding -> semantic dedup -> similarity -> persist -> reload.

    Uses real Ollama (nomic-embed-text) but no external network. Preference
    fixtures are written to tmp_path rather than read from data/likes.txt:
    those files are gitignored and personal, so a test reading them would fail
    on a fresh clone and change behaviour whenever the user edits a preference.
    """


    db_path = tmp_path / "smoke.db"
    init_db(db_path)
    logger = get_logger("smoke", stream=io.StringIO())

    likes = tmp_path / "likes.txt"
    likes.write_text("karaoke\npunk music\ncalm relaxed atmosphere\n\n[movies]\nhorror films\n")
    dislikes = tmp_path / "dislikes.txt"
    dislikes.write_text("bars\nnightclubs\ndancing\npop music\n")

    class CountingProvider:
        def __init__(self, inner):
            self.inner, self.calls = inner, 0

        def embed(self, text: str) -> list[float]:
            self.calls += 1
            return self.inner.embed(text)

    provider = CountingProvider(
        OllamaEmbeddingProvider(client=OllamaClient("http://localhost:11434", timeout=180))
    )
    repo = PreferenceRepository(provider, db_path, logger)

    # Cold cache: every preference line is embedded.
    prefs = repo.load(likes, dislikes)
    assert provider.calls == 8, f"expected 8 embeddings, got {provider.calls}"
    assert len(prefs.likes) == 4 and len(prefs.dislikes) == 4

    # Warm cache: unchanged files must not reach Ollama at all.
    provider.calls = 0
    prefs = repo.load(likes, dislikes)
    assert provider.calls == 0, "unchanged preference files must not be re-embedded"

    # Domain scoping: the movies preference is invisible to a general event.
    assert "horror films" not in [p.text for p in prefs.likes_for("general")]
    assert "horror films" in [p.text for p in prefs.likes_for("movies")]

    now = datetime(2026, 8, 6, 20, 0, tzinfo=timezone.utc)

    def event(event_id, tags, summary, venue, start=now):
        return Event(
            event_id=event_id, source_event_candidates=[event_id], source_type="apify",
            created_at=now, updated_at=now, venue=venue, start_time=start,
            title=event_id, summary=summary,
            tags=[Tag(text=t, weight=w) for t, w in tags],
        )

    events = [
        event("karaoke-night", [("karaoke", 1.0), ("singing", 0.8), ("bar", 0.2)],
              "a karaoke night with a full bar and food", "Koto"),
        event("sports-trivia", [("trivia night", 0.9), ("sports bar", 0.7), ("draft beer", 0.5)],
              "a pub trivia night with drink specials and sports on the screens", "O'Doul's"),
        # Same venue and night as karaoke-night, worded differently by another source.
        event("karaoke-dup", [("karaoke", 1.0), ("singing", 0.8), ("bar", 0.2)],
              "a karaoke night with a full bar and food", "Koto"),
        # Same event text a week later — a recurrence that must survive dedup.
        event("karaoke-next-week", [("karaoke", 1.0), ("singing", 0.8), ("bar", 0.2)],
              "a karaoke night with a full bar and food", "Koto", now + timedelta(days=7)),
    ]

    EmbeddingStage(provider, logger).process(events)

    for e in events:
        assert len(e.tag_embeddings) == len(e.tags)
        assert all(isinstance(b, bytes) for b in e.tag_embeddings)
        assert len(decode_vector(e.tag_embeddings[0])) == 768
        assert e.summary_embedding is not None

    deduped = SemanticDeduplicationEngine().deduplicate(events, DeduplicationConfig())
    ids = {e.event_id for e in deduped}
    assert "karaoke-next-week" in ids, "a weekly recurrence must not be merged away"
    assert len(deduped) >= 3

    SimilarityStage(preferences=prefs, config=ScoringConfig()).process(deduped)

    scores = {e.event_id: e.similarity.base_score for e in deduped}
    karaoke = next(e for e in deduped if e.event_id.startswith("karaoke"))
    trivia = next(e for e in deduped if e.event_id == "sports-trivia")

    assert karaoke.similarity.base_score > trivia.similarity.base_score, scores
    assert karaoke.similarity.base_score > 0, "karaoke is a stated like"
    assert trivia.similarity.base_score < 0, "a sports bar hits two dislikes"
    assert trivia.similarity.match == "no"
    assert karaoke.similarity.match != "no", "a strong like must survive an incidental bar tag"

    top = max(karaoke.similarity.reasons, key=lambda r: abs(r.contribution))
    assert top.factor == "like_similarity"
    assert top.matched_preference == "karaoke"

    # Persist and reload: the expensive work must survive the process.
    save_events(deduped, db_path)
    reloaded = {e.event_id: e for e in load_events(db_path)}
    assert set(reloaded) == ids

    original = karaoke
    restored = reloaded[original.event_id]
    assert restored.tags == original.tags
    assert restored.tag_embeddings == original.tag_embeddings
    assert restored.summary_embedding == original.summary_embedding
    assert restored.start_time == original.start_time

    # Reloaded events are already embedded, so a second pass costs nothing.
    provider.calls = 0
    EmbeddingStage(provider, logger).process(list(reloaded.values()))
    assert provider.calls == 0, "reloaded events must not be re-embedded"


def test_weather_comfort_smoke(tmp_path: Path) -> None:
    """Weather comfort end to end: shipped config, real SQLite, real persistence.

    Scores the same event under three conditions and checks the ordering the
    ranking engine will depend on, using the curves in config.example.yaml
    rather than test-local numbers.
    """


    # Load the shipped weather block through the real public entry point, so the
    # smoke test exercises the same path production does.
    example = yaml.safe_load(Path("config/config.example.yaml").read_text())
    example["location"] = {
        "latitude": 42.52,
        "longitude": -70.89,
        "postal_code": "01970",
        "search_radius_miles": 10,
    }
    config_file = tmp_path / "example_config.yaml"
    config_file.write_text(yaml.dump(example))
    weather_cfg = load_config(config_path=config_file).weather

    run_date = date(2025, 6, 21)
    now = datetime(2025, 6, 21, 12, 0, tzinfo=timezone.utc)
    local_tz = zoneinfo.ZoneInfo("America/New_York")
    start = datetime(2025, 6, 22, 20, 0, tzinfo=local_tz)

    def day(temp_f: float, dew_f: float, wind: float, precip: float, condition: str) -> dict:
        return {
            "date": "2025-06-22",
            "hours": [
                {
                    "hour": hour,
                    "temperature_f": temp_f,
                    "relative_humidity": 50.0,
                    "dew_point_f": dew_f,
                    "precipitation_mm": precip,
                    "wind_speed_mph": wind,
                    "condition": condition,
                }
                for hour in range(24)
            ],
        }

    conditions = {
        "crisp": day(45.0, 35.0, 4.0, 0.0, "clear"),
        # 1.5mm sits between the ideal trace and the zero bound, so rain shows
        # as gradation rather than the all-or-nothing the condition code gives.
        "drizzle": day(58.0, 50.0, 6.0, 1.5, "rain"),
        "muggy_storm": day(88.0, 74.0, 18.0, 9.0, "thunderstorm"),
    }

    cfg = AppConfig(
        location=LocationConfig(42.52, -70.89, "01970", 10.0, "America/New_York"),
        scraping=ScrapingConfig(),
        venue_discovery=VenueDiscoveryConfig(),
        weather=weather_cfg,
    )

    scores: dict[str, float] = {}
    for name, weather_day in conditions.items():
        db_path = tmp_path / f"comfort_{name}.db"
        init_db(db_path)

        provider = MagicMock(spec=WeatherProvider)
        provider.fetch.return_value = weather_day
        svc = EnrichmentService(
            weather_provider=provider,
            movie_provider=None,
            astronomical_calculator=AstronomicalCalculator(),
            synthetic_rules=[],
            config=cfg,
            db_path=db_path,
            weather_cache=SqliteWeatherCache(db_path),
            get_now=lambda: now,
        )

        event = Event(
            event_id=f"comfort-{name}",
            source_event_candidates=[],
            source_type="instagram",
            created_at=now,
            updated_at=now,
            title="Rooftop Set",
            setting="outdoor",
            start_time=start,
        )
        enriched = svc.enrich([event], run_date)[0]

        # Survives a real round trip through SQLite.
        save_events([enriched], db_path)
        reloaded = load_events(db_path)[0]
        assert reloaded.setting == "outdoor"
        assert reloaded.weather["sampled_hour"] == 20
        assert len(reloaded.weather["forecast"]["day_series"]) == 24
        assert reloaded.weather["observed"] is None

        scores[name] = compute_comfort(
            reloaded.weather["forecast"]["hour"], weather_cfg
        ).adjustment

    assert scores["crisp"] == pytest.approx(weather_cfg.max_positive_adjustment)
    assert 0 < scores["drizzle"] < scores["crisp"], "a drizzle at 58F is still a fine night out"
    assert scores["muggy_storm"] == pytest.approx(-weather_cfg.max_negative_adjustment)
    assert scores["crisp"] > scores["drizzle"] > scores["muggy_storm"]


@pytest.mark.integration
def test_ranking_smoke(tmp_path: Path) -> None:
    """Embeddings -> similarity -> ranking -> persist -> reload.

    Uses real Ollama embeddings, real SQLite, and the thresholds, multipliers
    and comfort curves shipped in config.example.yaml, so the ordering under
    test is the one production would produce.
    """

    example = yaml.safe_load(Path("config/config.example.yaml").read_text())
    example["location"] = {
        "latitude": 42.52,
        "longitude": -70.89,
        "postal_code": "01970",
        "search_radius_miles": 10,
    }
    config_file = tmp_path / "example_config.yaml"
    config_file.write_text(yaml.dump(example))
    config = load_config(config_path=config_file)

    db_path = tmp_path / "ranking_smoke.db"
    init_db(db_path)
    logger = get_logger("smoke", stream=io.StringIO())

    likes = tmp_path / "likes.txt"
    likes.write_text("karaoke\npunk music\nlive jazz\nquiet cafes\n")
    dislikes = tmp_path / "dislikes.txt"
    dislikes.write_text("nightclubs\nsports bars\ndancing\n")

    provider = OllamaEmbeddingProvider(client=OllamaClient("http://localhost:11434", timeout=180))
    prefs = PreferenceRepository(provider, db_path, logger).load(likes, dislikes)

    run_date = date(2026, 8, 6)
    now = datetime(2026, 8, 6, 20, 0, tzinfo=timezone.utc)

    def weather(temp_f: float, dew_f: float, wind: float, precip: float, condition: str) -> dict:
        hour = {
            "hour": 20,
            "temperature_f": temp_f,
            "dew_point_f": dew_f,
            "wind_speed_mph": wind,
            "precipitation_mm": precip,
            "condition": condition,
        }
        return {
            "sampled_hour": 20,
            "forecast": {"issued_at": now.isoformat(), "hour": hour, "day_series": []},
            "observed": None,
        }

    CLEAR = weather(62.0, 50.0, 5.0, 0.0, "clear")
    STORM = weather(88.0, 74.0, 18.0, 9.0, "thunderstorm")

    def event(event_id, tags, summary, venue, setting="indoor", weather_record=None,
              source="apify", description=None):
        return Event(
            event_id=event_id, source_event_candidates=[event_id], source_type=source,
            created_at=now, updated_at=now, venue=venue, start_time=now,
            title=event_id, description=description, summary=summary,
            setting=setting, weather=weather_record,
            tags=[Tag(text=t, weight=w) for t, w in tags],
        )

    karaoke_tags = [
        ("karaoke", 1.0), ("singing", 0.8), ("live music", 0.6),
        ("drinks", 0.3), ("thursday", 0.1),
    ]
    jazz_tags = [
        ("live jazz", 1.0), ("jazz quartet", 0.9), ("live music", 0.7),
        ("cocktails", 0.3), ("rooftop", 0.2),
    ]

    events = [
        event("karaoke-night", karaoke_tags, "a karaoke night with a full bar", "Koto"),
        # Same event, one surviving tag from a description long enough to have
        # earned several: a thin extraction, not a terse source.
        event("karaoke-thin", [("karaoke", 1.0)], "a karaoke night with a full bar", "Koto",
              description="A karaoke night with a full bar. " * 30),
        event("punk-show", [
            ("punk music", 1.0), ("live band", 0.9), ("loud", 0.5),
            ("all ages", 0.3), ("friday", 0.1),
        ], "a punk gig with three local bands", "The Basement"),
        event("quiet-cafe", [
            ("quiet cafe", 1.0), ("coffee", 0.8), ("reading", 0.5),
            ("pastries", 0.3), ("morning", 0.1),
        ], "a quiet cafe with good coffee and no music", "Bean There"),
        event("nightclub", [
            ("nightclub", 1.0), ("dancing", 0.9), ("dj set", 0.7),
            ("late night", 0.4), ("saturday", 0.1),
        ], "a late night club with a resident dj", "Pulse"),
        event("sports-trivia", [
            ("sports bar", 1.0), ("trivia night", 0.8), ("draft beer", 0.5),
            ("wings", 0.3), ("tuesday", 0.1),
        ], "a pub trivia night with sports on every screen", "O'Doul's"),
        # Identical events, opposite nights: weather is the only difference.
        event("rooftop-jazz-clear", jazz_tags, "a rooftop jazz set under the stars",
              "Sky Bar", setting="outdoor", weather_record=CLEAR),
        event("rooftop-jazz-storm", jazz_tags, "a rooftop jazz set under the stars",
              "Roof Garden", setting="outdoor", weather_record=STORM),
        event("no-venue-karaoke", karaoke_tags, "a karaoke night with a full bar", None),
        event("blocked-bar", karaoke_tags, "a karaoke night with a full bar", "The Sports Bar"),
    ]

    EmbeddingStage(provider, logger).process(events)
    SimilarityStage(preferences=prefs, config=config.scoring).process(events)

    blocklist = ["The Sports Bar"]
    engine = RankingEngine(config, blocklist=blocklist, logger=logger)
    scored, ranked = engine.rank(events, run_date)

    by_id = {s.event_id: s for s in scored}
    placed = {r.event_id: r for r in ranked}

    # Only the blocklisted venue is dropped; everything else survives, negatives included.
    assert "blocked-bar" not in by_id
    assert set(by_id) == {e.event_id for e in events} - {"blocked-bar"}
    assert "no-venue-karaoke" in by_id, "an event with no venue must not match a name entry"

    # Ranks are contiguous over what remains, and the list is in rank order.
    assert [r.rank for r in ranked] == list(range(1, len(ranked) + 1))
    assert ranked == sorted(ranked, key=lambda r: (-r.final_score, r.event_id))

    scores = {r.event_id: r.final_score for r in ranked}

    # Stated preferences decide the ordering.
    assert scores["karaoke-night"] > scores["nightclub"], scores
    assert scores["punk-show"] > scores["sports-trivia"], scores
    assert scores["nightclub"] < 0, "a nightclub hits two dislikes"

    # Weather separates two otherwise identical outdoor events.
    assert scores["rooftop-jazz-clear"] > scores["rooftop-jazz-storm"], scores
    assert placed["rooftop-jazz-clear"].weather_adjustment > 0
    assert placed["rooftop-jazz-storm"].weather_adjustment < 0

    # A thin extraction is uncertain, not bad: it sinks toward the middle.
    assert by_id["karaoke-thin"].tag_confidence < 1.0
    assert by_id["karaoke-night"].tag_confidence == 1.0
    assert scores["karaoke-thin"] < scores["karaoke-night"], scores

    # Every score component is explained.
    for score in scored:
        factors = {reason.factor for reason in score.reasons}
        assert "match_classification" in factors
        if placed[score.event_id].weather_adjustment != 0:
            assert "weather_adjustment" in factors
    assert "low_tag_confidence" in {r.factor for r in by_id["karaoke-thin"].reasons}

    # The order is the product: ranks run 1..n, descending by score.
    assert [r.rank for r in ranked] == list(range(1, len(ranked) + 1))
    scores = [r.final_score for r in ranked]
    assert scores == sorted(scores, reverse=True)

    # Re-ranking the same batch is identical, whatever order the events arrive in.
    shuffled = list(events)
    random.Random(7).shuffle(shuffled)
    assert engine.rank(shuffled, run_date) == (scored, ranked)

    # The run survives a real round trip through SQLite. The events go in first
    # because a score references the event it scored: persisting a verdict about
    # a row that was never stored is the thing the foreign key exists to reject.
    save_events(events, db_path)
    score_repo, ranking_repo = SqliteScoreRepository(db_path), SqliteRankingRepository(db_path)
    score_repo.save(scored)
    ranking_repo.save(ranked)
    assert {s.event_id for s in score_repo.for_run(run_date)} == set(by_id)
    assert ranking_repo.for_run(run_date) == ranked

    # A re-run supersedes its earlier attempt rather than accumulating a second copy.
    again_scored, again_ranked = engine.rank(shuffled, run_date)
    score_repo.save(again_scored)
    ranking_repo.save(again_ranked)
    assert ranking_repo.for_run(run_date) == ranked


# ---------------------------------------------------------------------------
# Phase 10 — CLI interface
# ---------------------------------------------------------------------------


def test_cli_smoke(tmp_path: Path) -> None:
    """Ranking -> persist -> CLI render, over ten events across two days.

    Not marked slow: the CLI's whole premise is that query time needs no model
    and no network, so a smoke test that needed either would be testing the
    wrong system. Similarity is attached directly here for the same reason —
    `RankingEngine` consumes it rather than re-scoring, and Phase 7's smoke
    already covers producing it.
    """

    example = yaml.safe_load(Path("config/config.example.yaml").read_text())
    example["location"] = {
        "latitude": 42.52,
        "longitude": -70.89,
        "postal_code": "01970",
        "search_radius_miles": 10,
    }
    config_file = tmp_path / "example_config.yaml"
    config_file.write_text(yaml.dump(example))
    config = load_config(config_path=config_file)

    db_path = tmp_path / "cli_smoke.db"
    init_db(db_path)

    tz = zoneinfo.ZoneInfo(config.location.timezone)
    run_date = date(2025, 6, 21)
    now = datetime(2025, 6, 21, 17, 0, tzinfo=tz)
    sunset = datetime(2025, 6, 21, 20, 15, tzinfo=tz)

    def _event(event_id: str, title: str, base: float, hour: int | None, day: int = 21) -> Event:
        start = datetime(2025, 6, day, hour, 0, tzinfo=tz) if hour is not None else None
        return Event(
            event_id=event_id,
            source_event_candidates=[f"cand-{event_id}"],
            source_type="instagram",
            created_at=now,
            updated_at=now,
            title=title,
            venue="The Dive Bar",
            start_time=start,
            tags=[Tag(text="karaoke", weight=1.0), Tag(text="live music", weight=0.8)],
            summary=f"{title} at The Dive Bar",
            astronomical_data={"sunset": sunset.isoformat()},
            similarity=SimilarityResult(
                tag_score=base,
                summary_score=base,
                base_score=base,
                match="yes" if base > 0.2 else "maybe",
                reasons=[
                    Reason(
                        factor="like_similarity",
                        matched_preference="karaoke night",
                        similarity=0.87,
                        contribution=base,
                        direction="positive" if base >= 0 else "negative",
                        tag="karaoke",
                    )
                ],
            ),
        )

    # Ten events over two days: eight tonight (one of them undated), two tomorrow.
    events = [
        _event("t1", "Karaoke Night", 0.62, 20),
        _event("t2", "Open Mic", 0.55, 19),
        _event("t3", "Late Jazz", 0.48, 22),
        _event("t4", "Afternoon Market", 0.20, 14),
        _event("t5", "Evening Talk", 0.12, 18),
        _event("t6", "Corporate Mixer", 0.01, 18),
        _event("t7", "Crypto Meetup", -0.35, 19),
        _event("u1", "Open Studio Weekend", 0.40, None),
        _event("m1", "Tomorrow Gig", 0.58, 20, day=22),
        _event("m2", "Tomorrow Matinee", 0.30, 11, day=22),
    ]

    engine = RankingEngine(config, blocklist=[], logger=get_logger("smoke", stream=io.StringIO()))
    scored, ranked = engine.rank(events, run_date)
    assert len(ranked) == 10

    save_events(events, db_path)
    SqliteScoreRepository(db_path).save(scored)
    SqliteRankingRepository(db_path).save(ranked)

    def _invoke(*argv: str) -> str:
        stdout = io.StringIO()
        code = run(
            ["--db", str(db_path), *argv],
            get_now=lambda: now,
            stdout=stdout,
            stderr=io.StringIO(),
        )
        assert code == 0
        return stdout.getvalue()

    started = time.perf_counter()
    default_view = _invoke()
    elapsed = time.perf_counter() - started

    assert elapsed < 1.0, f"the CLI took {elapsed:.3f}s reading precomputed rows"

    # Only tonight, plus the undated event ranked among the rest.
    assert "Karaoke Night" in default_view
    assert "Tomorrow Gig" not in default_view
    assert "Tomorrow Matinee" not in default_view
    assert "Open Studio Weekend" in default_view
    assert "time TBC" in default_view

    # The engine's order survives to the screen, and its reasons with it.
    assert default_view.index("Karaoke Night") < default_view.index("Open Mic")
    assert "karaoke night" in default_view

    # Nothing is lost: what the view cuts, it counts, and --all shows.
    # The count is checked against the view itself, not against `ranked` — the
    # night window filters before the limit applies, so the two differ.
    limited = _invoke("--limit", "2")
    everything = _invoke("--all")
    listed = re.findall(r"^  \d+\. .*$", limited, re.M)
    assert len(listed) == 2

    hidden = re.search(r"\+ (\d+) more events? ranked lower", limited)
    assert hidden, "a cut list must always say how much it cut"
    assert len(re.findall(r"^  \d+\. .*$", everything, re.M)) == 2 + int(hidden.group(1))
    assert "ranked lower (--all)" not in everything

    # Raw ignores ranking entirely and shows every stored event.
    raw_view = _invoke("--raw")
    for event in events:
        assert event.title in raw_view
    assert "TOP PICKS" not in raw_view

    # Filters compose and stay honest about timing they cannot assert.
    windowed = _invoke("--time", "19:30-23:00")
    assert "Late Jazz" in windowed
    assert "Afternoon Market" not in windowed
    assert "Open Studio Weekend" not in windowed

    after_dark = _invoke("--after-sunset")
    assert "Late Jazz" in after_dark
    assert "Open Mic" not in after_dark

    # Seed management writes where it is told and does not duplicate.
    seeds = tmp_path / "seeds.yaml"
    _invoke("add-source", "@smoketest", "--seeds-file", str(seeds))
    _invoke("add-source", "@smoketest", "--seeds-file", str(seeds))
    assert yaml.safe_load(seeds.read_text())["handles"] == ["@smoketest"]


# ---------------------------------------------------------------------------
# Batch orchestrator
# ---------------------------------------------------------------------------


ICS_FIXTURE = Path("tests/fixtures/northshorenightout.ics")
HTML_FIXTURE = Path("tests/fixtures/northshorenightout.html")
#: Both fixtures were captured on 2026-08-05, so their contents resolve
#: against that day.
BATCH_NOW = datetime(2026, 8, 5, 12, 0, tzinfo=timezone.utc)


class _FixtureSession:
    """Serves one captured document, counting how often it is fetched."""

    def __init__(self, body: str) -> None:
        self._body = body
        self.calls = 0

    def get(self, url, headers=None, timeout=None):
        self.calls += 1
        return MagicMock(status_code=200, headers={}, text=self._body, raise_for_status=lambda: None)


class _StubWeather(WeatherProvider):
    """A clear 70F day for whatever date is asked for."""

    def fetch(self, date, lat, lng):
        return {
            "date": date.isoformat(),
            "hours": [
                {
                    "hour": hour,
                    "temperature_f": 70.0,
                    "relative_humidity": 45.0,
                    "dew_point_f": 48.0,
                    "precipitation_mm": 0.0,
                    "wind_speed_mph": 5.0,
                    "condition": "clear",
                }
                for hour in range(24)
            ],
        }


class _CountingExtraction:
    """Stands in for LLM Pass 1, recording every text it was asked to extract."""

    def __init__(self) -> None:
        self.texts: list[str] = []

    def extract(self, text, image_bytes=None, reference_date=None):
        self.texts.append(text)
        return ExtractionResult(
            title=None,
            venue=None,
            start_time=None,
            end_time=None,
            tags=[Tag(text="live music", weight=1.0), Tag(text="bar", weight=0.4)],
            summary="An evening of live music at a local bar.",
            model="fake-extraction-model",
            prompt_version="fakever0",
            degradation=None,
            setting="indoor",
        )


class _StubEmbeddings:
    """Deterministic vectors, so ordering is stable without Ollama."""

    def __init__(self) -> None:
        self.calls = 0

    def embed(self, text: str) -> list[float]:
        self.calls += 1
        rng = random.Random(text)
        return [rng.uniform(-1.0, 1.0) for _ in range(16)]


def _batch_config(tmp_path: Path) -> AppConfig:
    example = yaml.safe_load(Path("config/config.example.yaml").read_text())
    example["location"] = {
        "latitude": 42.52,
        "longitude": -70.89,
        "postal_code": "01970",
        "search_radius_miles": 10,
    }
    config_file = tmp_path / "batch_config.yaml"
    config_file.write_text(yaml.dump(example))
    return load_config(config_path=config_file)


def _batch_dependencies(tmp_path: Path, db_path: Path, config: AppConfig, logger):
    """Wire the real pipeline over both fixtures, faking only what leaves the box."""
    seeds = tmp_path / "seeds.yaml"
    seeds.write_text(yaml.dump({"handles": [], "venues": []}))
    likes = tmp_path / "likes.txt"
    likes.write_text("live music\nkaraoke\n")
    dislikes = tmp_path / "dislikes.txt"
    dislikes.write_text("nightclubs\n")

    ics = IcsCalendarSource(
        config=FeedConfig("nsno_cal", "https://cal.example.com/basic.ics", "nsno_cal"),
        http_cache=InMemoryHttpCache(),
        session=_FixtureSession(ICS_FIXTURE.read_text(encoding="utf-8")),
        get_now=lambda: BATCH_NOW,
        logger=logger,
    )
    html = HtmlListingSource(
        config=FeedConfig("nsno_list", "https://listings.example.com/", "nsno_list"),
        http_cache=InMemoryHttpCache(),
        tzname="America/New_York",
        session=_FixtureSession(HTML_FIXTURE.read_text(encoding="utf-8")),
        get_now=lambda: BATCH_NOW,
        logger=logger,
    )

    embeddings = _StubEmbeddings()
    extraction = _CountingExtraction()

    return {
        "ingestion_service": IngestionService(
            config=config,
            db_path=db_path,
            entities=SqliteEntityRepository(db_path),
            seeds_path=seeds,
            failover_sources=[],
            independent_sources=[ics, html],
            logger=logger,
        ),
        "normalization_service": NormalizationService(config, logger),
        "enrichment_service": EnrichmentService(
            weather_provider=_StubWeather(),
            movie_provider=None,
            astronomical_calculator=AstronomicalCalculator(),
            synthetic_rules=[],
            config=config,
            db_path=db_path,
            weather_cache=SqliteWeatherCache(db_path),
            get_now=lambda: BATCH_NOW,
            logger=logger,
        ),
        "extraction_stage": ExtractionStage(
            provider=extraction,
            image_fetcher=None,
            logger=logger,
            get_now=lambda: BATCH_NOW,
        ),
        "embedding_stage": EmbeddingStage(embeddings, logger),
        "semantic_deduplicator": SemanticDeduplicationEngine(),
        "similarity_stage": SimilarityStage(
            PreferenceRepository(embeddings, db_path, logger).load(likes, dislikes),
            config.scoring,
        ),
        "ranking_engine": RankingEngine(config, [], logger),
    }, extraction


def test_batch_smoke(tmp_path: Path) -> None:
    """The whole batch over both captured feeds, then again to prove it is incremental.

    Only what leaves the box is faked: the HTTP sessions serve fixtures, and the
    LLM and weather providers are stubs. Everything between them is the real
    pipeline, running through real SQLite.
    """
    config = _batch_config(tmp_path)
    db_path = tmp_path / "batch_smoke.db"
    init_db(db_path)
    logger = get_logger("batch_smoke", stream=io.StringIO())
    run_date = date(2026, 8, 5)

    deps, extraction = _batch_dependencies(tmp_path, db_path, config, logger)
    result = run_batch(
        candidate_repository=SqliteCandidateRepository(db_path),
        event_repository=SqliteEventRepository(db_path),
        run_repository=SqliteRunRepository(db_path),
        score_repository=SqliteScoreRepository(db_path),
        ranking_repository=SqliteRankingRepository(db_path),
        config=config,
        db_path=db_path,
        logger=logger,
        run_date=run_date,
        get_now=lambda: BATCH_NOW,
        **deps,
    )

    assert result.outcome == "success", result.errors
    assert result.stage_counts["ingested"] > 0
    assert result.rankings, "the batch produced no rankings"

    first_pass_extractions = len(extraction.texts)
    assert first_pass_extractions > 0

    # Rankings reached the database, not just the return value.
    conn = sqlite3.connect(db_path)
    try:
        persisted = conn.execute("SELECT COUNT(*) FROM rankings").fetchone()[0]
        runs = conn.execute(
            "SELECT outcome, completed_at FROM run_history"
        ).fetchall()
    finally:
        conn.close()

    assert persisted == len(result.rankings)
    assert runs == [("success", BATCH_NOW.isoformat())]

    # --- second run, same night -------------------------------------------
    second_deps, second_extraction = _batch_dependencies(
        tmp_path, db_path, config, logger
    )
    second = run_batch(
        candidate_repository=SqliteCandidateRepository(db_path),
        event_repository=SqliteEventRepository(db_path),
        run_repository=SqliteRunRepository(db_path),
        score_repository=SqliteScoreRepository(db_path),
        ranking_repository=SqliteRankingRepository(db_path),
        config=config,
        db_path=db_path,
        logger=logger,
        run_date=run_date,
        get_now=lambda: BATCH_NOW,
        **second_deps,
    )

    assert second.outcome == "success", second.errors
    assert second_extraction.texts == [], (
        "the second run re-extracted; the save-after-extraction design is broken"
    )
    assert second.rankings
