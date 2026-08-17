"""Enrichment must not ask a forecast API for dates it cannot answer.

Measured on the live corpus before this existed: the ranked set spans **98
distinct dates**, of which only **24** ever landed in `weather_cache`. The other
74 returned nothing, and a `None` is not cached — so those 74 requests were
re-issued on every batch run, every night, and again on every read-time rescore.

That is the whole bug. It is not a slow rescore with a tidy fix; it is a third
party being asked 74 pointless questions on a schedule.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from src.config import (
    AppConfig,
    LocationConfig,
    ScoringConfig,
    ScrapingConfig,
    VenueDiscoveryConfig,
    WeatherConfig,
)
from src.enrichment.astronomical import AstronomicalCalculator
from src.enrichment.service import EnrichmentService
from src.models.event import Event
from src.storage.memory.weather_cache import InMemoryWeatherCache
from src.storage.sqlite.connection import init_db
from src.utils.logging import get_logger

TZ = timezone(timedelta(hours=-4))
TODAY = date(2026, 8, 17)
NOW = datetime(2026, 8, 17, 2, 0, tzinfo=TZ)


class _CountingWeather:
    """Records every date it was asked for. A boundary, so a fake belongs."""

    def __init__(self) -> None:
        self.asked: list[date] = []

    def fetch(self, day: date, lat: float, lng: float) -> dict[str, Any]:
        self.asked.append(day)
        return {
            "date": day.isoformat(),
            "hours": [{"hour": 20, "temperature_f": 66.0, "condition": "clear"}],
        }


def _config(**weather: Any) -> AppConfig:
    return AppConfig(
        location=LocationConfig(42.52, -70.89, "01970", 10.0, "America/New_York"),
        scraping=ScrapingConfig(),
        venue_discovery=VenueDiscoveryConfig(blocklist_name_match_threshold=0.80),
        scoring=ScoringConfig(),
        weather=WeatherConfig(air_quality_enabled=False, **weather),
    )


def _event(day: date) -> Event:
    return Event(
        event_id=f"evt-{day}",
        source_event_candidates=[],
        source_type="instagram",
        created_at=NOW,
        updated_at=NOW,
        title="Test Event",
        start_time=datetime(day.year, day.month, day.day, 20, 0, tzinfo=TZ),
    )


def _service(tmp_path: Path, provider: _CountingWeather, config: AppConfig):
    db = tmp_path / "weather.db"
    init_db(db)
    return EnrichmentService(
        weather_provider=provider,
        movie_provider=None,
        astronomical_calculator=AstronomicalCalculator(),
        synthetic_rules=[],
        config=config,
        db_path=db,
        weather_cache=InMemoryWeatherCache(),
        air_quality_provider=None,
        get_now=lambda: NOW,
        logger=get_logger("horizon_test"),
    )


def _asked_for(tmp_path, days: list[date], config: AppConfig | None = None) -> list[date]:
    """Which *event* dates the provider was asked for.

    `enrich` always asks for the run date as well, because synthetic activities
    are conditioned on tonight's weather and have no start time of their own.
    That request is unrelated to the horizon and is stripped here so the
    assertions below say what they mean — its presence is asserted separately.
    """
    provider = _CountingWeather()
    service = _service(tmp_path, provider, config or _config())
    service.enrich([_event(day) for day in days], TODAY)
    asked = list(provider.asked)
    if TODAY in asked and TODAY not in days:
        asked.remove(TODAY)
    return asked


class TestForecastHorizon:
    def test_a_date_inside_the_horizon_is_fetched(self, tmp_path):
        wanted = TODAY + timedelta(days=5)

        assert _asked_for(tmp_path, [wanted]) == [wanted]

    def test_today_is_fetched(self, tmp_path):
        assert _asked_for(tmp_path, [TODAY]) == [TODAY]

    def test_the_run_date_is_always_fetched_for_synthetic_activities(self, tmp_path):
        """They are conditioned on tonight's weather and have no start of their own.

        Asserted here because the helper above strips this request, and a
        stripped thing that stopped happening would go unnoticed.
        """
        provider = _CountingWeather()
        service = _service(tmp_path, provider, _config())

        service.enrich([], TODAY)

        assert provider.asked == [TODAY]

    def test_the_last_day_of_the_horizon_is_fetched(self, tmp_path):
        """Inclusive: the bound names a day the API answers, not the first it does not."""
        edge = TODAY + timedelta(days=16)

        assert _asked_for(tmp_path, [edge], _config(forecast_horizon_days=16)) == [edge]

    def test_a_date_beyond_the_horizon_is_never_asked_for(self, tmp_path):
        """Open-Meteo answers about sixteen days. Asking for day ninety is noise."""
        beyond = TODAY + timedelta(days=17)

        assert _asked_for(tmp_path, [beyond], _config(forecast_horizon_days=16)) == []

    def test_a_past_date_is_never_asked_for(self, tmp_path):
        """A forecast for a night that has been is not a thing to request."""
        assert _asked_for(tmp_path, [TODAY - timedelta(days=1)]) == []

    def test_the_horizon_is_configurable(self, tmp_path):
        near = TODAY + timedelta(days=3)
        far = TODAY + timedelta(days=10)

        asked = _asked_for(tmp_path, [near, far], _config(forecast_horizon_days=5))

        assert asked == [near]

    def test_a_ninety_day_listing_costs_a_horizon_of_requests(self, tmp_path):
        """The live shape, as one assertion.

        Ninety-eight dates went out and twenty-four came back. What goes out now
        is bounded by what can come back.
        """
        days = [TODAY + timedelta(days=offset) for offset in range(90)]

        asked = _asked_for(tmp_path, days, _config(forecast_horizon_days=16))

        assert len(asked) == 17, "one per day from today to the horizon, inclusive"
        assert max(asked) == TODAY + timedelta(days=16)


class TestUnchangedBehaviour:
    def test_an_event_inside_the_horizon_still_gets_its_weather(self, tmp_path):
        """The bound must not cost the events it was meant to leave alone."""
        provider = _CountingWeather()
        service = _service(tmp_path, provider, _config())
        events = [_event(TODAY + timedelta(days=2))]

        service.enrich(events, TODAY)

        assert events[0].weather is not None

    def test_an_event_beyond_the_horizon_simply_has_no_weather(self, tmp_path):
        """Which is what it had before, having asked and been told nothing.

        `weather_adjustment` already treats absent weather as *unknown* rather
        than bad, so nothing downstream changes — only the request stops.
        """
        provider = _CountingWeather()
        service = _service(tmp_path, provider, _config(forecast_horizon_days=16))
        events = [_event(TODAY + timedelta(days=40))]

        service.enrich(events, TODAY)

        assert events[0].weather is None

    def test_each_date_is_asked_for_once(self, tmp_path):
        """Two events on one night are one request, as before."""
        day = TODAY + timedelta(days=1)
        first, second = _event(day), _event(day)
        second.event_id = "evt-second"
        provider = _CountingWeather()
        service = _service(tmp_path, provider, _config())

        service.enrich([first, second], TODAY)

        assert provider.asked.count(day) == 1


class _CountingAirQuality:
    """Records every date it was asked for."""

    def __init__(self) -> None:
        self.asked: list[date] = []

    def fetch(self, day: date, lat: float, lng: float) -> dict[str, Any]:
        self.asked.append(day)
        return {"date": day.isoformat(), "hours": [{"hour": 20, "us_aqi": 20}]}


def _aqi_asked_for(tmp_path, days: list[date], config: AppConfig) -> list[date]:
    # The shared `_config` helper turns air quality off, because the weather
    # tests want one provider's calls and not two. These tests are about the
    # other provider, so it goes back on here — leaving it off made every
    # assertion below pass against a service that fetched nothing at all.
    config.weather.air_quality_enabled = True
    aqi = _CountingAirQuality()
    db = tmp_path / "weather.db"
    init_db(db)
    service = EnrichmentService(
        weather_provider=_CountingWeather(),
        movie_provider=None,
        astronomical_calculator=AstronomicalCalculator(),
        synthetic_rules=[],
        config=config,
        db_path=db,
        weather_cache=InMemoryWeatherCache(),
        air_quality_provider=aqi,
        get_now=lambda: NOW,
        logger=get_logger("aqi_horizon_test"),
    )
    service.enrich([_event(day) for day in days], TODAY)
    return aqi.asked


class TestAirQualityHorizon:
    """The worse half of the same bug, and a second endpoint.

    Air quality had no bound *and* no persistent cache — only a per-run dict —
    so every distinct date in a ninety-day listing became a request, every run,
    and none of them were remembered. Its own docstring already said its horizon
    is shorter than the weather forecast's.
    """

    def test_a_date_inside_the_air_quality_horizon_is_fetched(self, tmp_path):
        wanted = TODAY + timedelta(days=2)

        assert _aqi_asked_for(
            tmp_path, [wanted], _config(air_quality_horizon_days=5)
        ) == [wanted]

    def test_a_date_beyond_it_is_never_asked_for(self, tmp_path):
        beyond = TODAY + timedelta(days=6)

        assert _aqi_asked_for(tmp_path, [beyond], _config(air_quality_horizon_days=5)) == []

    def test_its_horizon_is_shorter_than_the_weather_one(self, tmp_path):
        """A separate key because it is a separate service with a separate range.

        Sharing the weather horizon would ask the air quality API for eleven days
        it cannot answer, which is the same mistake one layer down.
        """
        day = TODAY + timedelta(days=10)
        config = _config(forecast_horizon_days=16, air_quality_horizon_days=5)

        assert _asked_for(tmp_path, [day], config) == [day]
        assert _aqi_asked_for(tmp_path, [day], config) == []

    def test_a_ninety_day_listing_costs_a_handful_of_requests(self, tmp_path):
        days = [TODAY + timedelta(days=offset) for offset in range(90)]

        asked = _aqi_asked_for(tmp_path, days, _config(air_quality_horizon_days=5))

        assert len(asked) == 6, "one per day from today to the horizon, inclusive"
