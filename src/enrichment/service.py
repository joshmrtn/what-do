"""EnrichmentService — orchestrates weather, astronomical, movie, and synthetic enrichment."""

import json
import sqlite3
import uuid
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Callable

from src.config import AppConfig, SyntheticActivityRule
from src.enrichment.astronomical import AstronomicalCalculator, AstronomicalData
from src.enrichment.movies import MovieMetadataProvider, enrich_movie_event
from src.enrichment.synthetic import SyntheticActivityGenerator
from src.enrichment.weather import WeatherProvider
from src.models.event import Event
from src.utils.logging import StructuredLogger, get_logger


class EnrichmentService:
    """Orchestrates all enrichment steps: weather, astronomical, movie metadata, and synthetic activities."""

    def __init__(
        self,
        weather_provider: WeatherProvider,
        movie_provider: MovieMetadataProvider | None,
        astronomical_calculator: AstronomicalCalculator,
        synthetic_rules: list[SyntheticActivityRule],
        config: AppConfig,
        db_path: Path,
        get_now: Callable[[], datetime] = datetime.now,
        logger: StructuredLogger | None = None,
    ) -> None:
        self._weather_provider = weather_provider
        self._movie_provider = movie_provider
        self._calculator = astronomical_calculator
        self._synthetic_rules = synthetic_rules
        self._config = config
        self._db_path = db_path
        self._get_now = get_now
        self._logger = logger or get_logger("enrichment")
        self._generator = SyntheticActivityGenerator()

    def enrich(self, events: list[Event], run_date: date) -> list[Event]:
        """Enrich events with weather, astronomical, and movie data; append synthetic activities.

        Args:
            events: Normalized, deduplicated events to enrich.
            run_date: The batch run date (used for synthetic activity generation).

        Returns:
            Enriched events with synthetic activities appended at the end.
        """
        lat = self._config.location.latitude
        lng = self._config.location.longitude
        tzname = self._config.location.timezone

        # Per-run in-memory caches to avoid redundant DB reads within a single batch
        _weather_cache: dict[date, dict | None] = {}
        _astro_cache: dict[date, AstronomicalData] = {}

        for event in events:
            if event.start_time is None:
                continue

            event_date = event.start_time.date()

            # --- Astronomical data ---
            if event_date not in _astro_cache:
                _astro_cache[event_date] = self._calculator.calculate(event_date, lat, lng, tzname)
            astro = _astro_cache[event_date]
            event.astronomical_data = {
                "sunrise": astro.sunrise.isoformat(),
                "sunset": astro.sunset.isoformat(),
                "dawn": astro.dawn.isoformat(),
                "dusk": astro.dusk.isoformat(),
            }

            # --- Weather ---
            if event_date not in _weather_cache:
                _weather_cache[event_date] = self._fetch_weather(event_date, lat, lng)
            event.weather = _weather_cache[event_date]

        # --- Movie metadata ---
        if self._movie_provider is not None:
            for event in events:
                enrich_movie_event(event, self._movie_provider, self._logger)

        # --- Synthetic activities ---
        run_astro = self._calculator.calculate(run_date, lat, lng, tzname)
        run_weather = self._fetch_weather(run_date, lat, lng)
        synthetic = self._generator.generate(
            self._synthetic_rules,
            run_date,
            run_weather,
            run_astro,
            self._get_now,
        )

        return events + synthetic

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fetch_weather(self, event_date: date, lat: float, lng: float) -> dict | None:
        """Return weather for (date, lat, lng), using DB cache; on miss, fetch and cache."""
        # Check DB cache first
        cached = self._db_weather_get(event_date, lat, lng)
        if cached is not None:
            return cached

        # Cache miss — call provider
        try:
            weather = self._weather_provider.fetch(event_date, lat, lng)
        except Exception as exc:
            self._logger.error(
                f"Weather fetch failed for {event_date}: {exc}",
                component="enrichment",
            )
            return None

        if weather is not None:
            self._db_weather_put(event_date, lat, lng, weather)

        return weather

    def _db_weather_get(self, event_date: date, lat: float, lng: float) -> dict | None:
        with sqlite3.connect(self._db_path) as conn:
            row = conn.execute(
                "SELECT data FROM weather_cache WHERE date=? AND latitude=? AND longitude=?",
                (event_date.isoformat(), lat, lng),
            ).fetchone()
        return json.loads(row[0]) if row else None

    def _db_weather_put(self, event_date: date, lat: float, lng: float, weather: dict) -> None:
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                """INSERT OR REPLACE INTO weather_cache
                   (id, date, latitude, longitude, data, fetched_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    str(uuid.uuid4()),
                    event_date.isoformat(),
                    lat,
                    lng,
                    json.dumps(weather),
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
