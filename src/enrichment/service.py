"""EnrichmentService — orchestrates weather, astronomical, movie, and synthetic enrichment."""

import json
import sqlite3

from src.storage.db import connect
import uuid
from datetime import date, datetime
from pathlib import Path
from typing import Any, Callable

from src.config import AppConfig, SyntheticActivityRule
from src.enrichment.air_quality import AirQualityProvider
from src.enrichment.astronomical import AstronomicalCalculator, AstronomicalData
from src.enrichment.movies import MovieMetadataProvider, enrich_movie_event
from src.enrichment.synthetic import SyntheticActivityGenerator
from src.enrichment.weather import WeatherProvider, sample_hour
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
        air_quality_provider: AirQualityProvider | None = None,
        get_now: Callable[[], datetime] = datetime.now,
        logger: StructuredLogger | None = None,
    ) -> None:
        self._weather_provider = weather_provider
        self._air_quality_provider = air_quality_provider
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
        _weather_cache: dict[date, dict[str, Any] | None] = {}
        _aqi_cache: dict[date, dict[str, Any] | None] = {}
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
            if event_date not in _aqi_cache:
                _aqi_cache[event_date] = self._fetch_air_quality(event_date, lat, lng)
            event.weather = self._weather_for(
                _weather_cache[event_date],
                # A placed start is not a published one. Sampling 04:00 for an
                # all-day event would score it on the coldest, darkest hour of
                # its day; `default_hour` is the evening the rest of the system
                # judges an unknown time by.
                event.start_time if event.states_a_time else None,
                _aqi_cache[event_date],
            )

        # --- Movie metadata ---
        if self._movie_provider is not None:
            for event in events:
                enrich_movie_event(event, self._movie_provider, self._logger)

        # --- Synthetic activities ---
        run_astro = self._calculator.calculate(run_date, lat, lng, tzname)
        run_day = self._fetch_weather(run_date, lat, lng)
        # Synthetic rules have no start time of their own, so they are judged at
        # the configured default hour rather than the daily extreme.
        run_hour = (
            None
            if run_day is None
            else sample_hour(run_day, None, self._config.weather.default_hour)
        )
        synthetic = self._generator.generate(
            self._synthetic_rules,
            run_date,
            run_hour,
            run_astro,
            self._get_now,
        )

        return events + synthetic

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fetch_air_quality(
        self, event_date: date, lat: float, lng: float
    ) -> dict[str, Any] | None:
        """Return a day of air quality, or None if disabled, absent, or failing.

        Air quality is advisory: its forecast horizon is far shorter than the
        weather forecast, so a miss is the normal case and never an error.
        """
        if self._air_quality_provider is None or not self._config.weather.air_quality_enabled:
            return None
        try:
            return self._air_quality_provider.fetch(event_date, lat, lng)
        except Exception as exc:
            self._logger.error(
                f"Air quality fetch failed for {event_date}: {exc}",
                component="enrichment",
            )
            return None

    def _weather_for(
        self,
        day: dict[str, Any] | None,
        start_time: datetime | None,
        air_quality_day: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """Build the event's persisted weather record from a fetched day.

        Keeps the raw readings and the full day series rather than a derived
        score, so retuned comfort curves can rescore history without a refetch,
        and a future cache purge cannot orphan an event from its conditions.
        The `observed` slot is reserved for a later backfill of what actually
        happened — a forecast issued days out is not what the user experienced.

        Returns:
            The persisted weather dict, or None if the day or hour is unavailable.
        """
        if day is None:
            return None
        hour = sample_hour(day, start_time, self._config.weather.default_hour)
        if hour is None:
            return None

        if air_quality_day is not None:
            aqi_hour = sample_hour(air_quality_day, start_time, self._config.weather.default_hour)
            if aqi_hour is not None:
                # A missing reading is left absent rather than written as None,
                # so comfort drops the factor and renormalises instead of
                # scoring unknown air as mediocre air.
                hour = {
                    **hour,
                    **{k: v for k, v in aqi_hour.items() if k != "hour" and v is not None},
                }

        return {
            "sampled_hour": hour["hour"],
            "forecast": {
                "issued_at": self._get_now().isoformat(),
                "hour": hour,
                "day_series": day.get("hours", []),
            },
            "observed": None,
        }

    def _fetch_weather(
        self, event_date: date, lat: float, lng: float
    ) -> dict[str, Any] | None:
        """Return weather for (date, lat, lng), using DB cache; on miss, fetch and cache.

        A cached day is only served while it is still fresh. An event discovered
        days ahead would otherwise be scored forever against the forecast issued
        on the day it was found, including on the night it actually happens.
        """
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

    def _db_weather_get(
        self, event_date: date, lat: float, lng: float
    ) -> dict[str, Any] | None:
        """Return the cached day, or None if absent or past its TTL."""
        with connect(self._db_path) as conn:
            row = conn.execute(
                """SELECT data, fetched_at FROM weather_cache
                   WHERE date=? AND latitude=? AND longitude=?""",
                (event_date.isoformat(), lat, lng),
            ).fetchone()
        if row is None or not self._is_fresh(row[1]):
            return None
        cached: dict[str, Any] = json.loads(row[0])
        return cached

    def _is_fresh(self, fetched_at: str) -> bool:
        """Whether a cache entry stamped `fetched_at` may still be served."""
        try:
            stamped = datetime.fromisoformat(fetched_at)
        except ValueError:
            # An unparseable stamp is treated as expired: refetching costs one
            # request, while trusting it could serve a forecast of any age.
            return False

        now = self._get_now()
        # The stamp is written by this same clock, so a mismatch means the clock
        # itself changed shape. Compare on common ground rather than raising.
        if (stamped.tzinfo is None) != (now.tzinfo is None):
            stamped = stamped.replace(tzinfo=now.tzinfo)

        age_hours = (now - stamped).total_seconds() / 3600
        return age_hours <= self._config.weather.cache_ttl_hours

    def _db_weather_put(
        self, event_date: date, lat: float, lng: float, weather: dict[str, Any]
    ) -> None:
        with connect(self._db_path) as conn:
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
                    self._get_now().isoformat(),
                ),
            )
