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
    from src.models.tag import Tag
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


@pytest.mark.slow
def test_semantic_matching_smoke(tmp_path: Path) -> None:
    """Embedding -> semantic dedup -> similarity -> persist -> reload.

    Uses real Ollama (nomic-embed-text) but no external network. Preference
    fixtures are written to tmp_path rather than read from data/likes.txt:
    those files are gitignored and personal, so a test reading them would fail
    on a fresh clone and change behaviour whenever the user edits a preference.
    """
    import io
    from datetime import datetime, timedelta, timezone

    from src.config import DeduplicationConfig, ScoringConfig
    from src.models.event import Event
    from src.models.tag import Tag
    from src.normalization.semantic_dedup import SemanticDeduplicationEngine
    from src.scoring.embedding_stage import EmbeddingStage
    from src.scoring.embeddings import OllamaEmbeddingProvider
    from src.scoring.preferences import PreferenceRepository
    from src.scoring.similarity_stage import SimilarityStage
    from src.storage.db import init_db
    from src.storage.events import load_events, save_events
    from src.utils.logging import get_logger
    from src.utils.ollama_client import OllamaClient
    from src.utils.vectors import decode_vector

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
    import zoneinfo
    from datetime import date, datetime, timezone
    from unittest.mock import MagicMock

    from src.config import (
        AppConfig,
        LocationConfig,
        ScrapingConfig,
        VenueDiscoveryConfig,
        _load_weather,
    )
    from src.enrichment.astronomical import AstronomicalCalculator
    from src.enrichment.comfort import compute_comfort
    from src.enrichment.service import EnrichmentService
    from src.enrichment.weather import WeatherProvider
    from src.models.event import Event
    from src.storage.db import init_db
    from src.storage.events import load_events, save_events

    example = yaml.safe_load(Path("config/config.example.yaml").read_text())
    weather_cfg = _load_weather(example["weather"])

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
