"""Unit tests for EnrichmentService."""

import json
import sqlite3
import zoneinfo
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.config import (
    AppConfig,
    LocationConfig,
    ScrapingConfig,
    SyntheticActivityRule,
    SyntheticConditions,
    VenueDiscoveryConfig,
)
from src.enrichment.astronomical import AstronomicalCalculator
from src.enrichment.movies import MovieMetadataProvider
from src.enrichment.service import EnrichmentService
from src.enrichment.air_quality import AirQualityProvider
from src.enrichment.weather import WeatherProvider
from src.models.event import Event
from src.storage.db import init_db
from src.utils.logging import StructuredLogger

# ---------------------------------------------------------------------------
# Constants & helpers
# ---------------------------------------------------------------------------

TZ = "America/New_York"
SALEM_LAT = 42.52
SALEM_LNG = -70.89
RUN_DATE = date(2025, 6, 21)
NOW = datetime(2025, 6, 21, 12, 0, tzinfo=timezone.utc)
LOCAL_TZ = zoneinfo.ZoneInfo(TZ)

def _hour_record(hour: int, condition: str = "clear") -> dict:
    """One hour of readings. Temperature varies by hour so tests can prove which was sampled."""
    return {
        "hour": hour,
        "temperature_f": 60.0 + hour,
        "relative_humidity": 40.0,
        "dew_point_f": 50.0,
        "precipitation_mm": 0.0,
        "wind_speed_mph": 5.0,
        "condition": condition,
    }


CLEAR_DAY = {"date": "2025-06-21", "hours": [_hour_record(h) for h in range(24)]}


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    p = tmp_path / "test.db"
    init_db(p)
    return p


@pytest.fixture
def cfg() -> AppConfig:
    return AppConfig(
        location=LocationConfig(SALEM_LAT, SALEM_LNG, "01970", 10.0, TZ),
        scraping=ScrapingConfig(),
        venue_discovery=VenueDiscoveryConfig(),
    )


def _weather_provider(return_value=CLEAR_DAY, side_effect=None) -> WeatherProvider:
    p = MagicMock(spec=WeatherProvider)
    if side_effect is not None:
        p.fetch.side_effect = side_effect
    else:
        p.fetch.return_value = return_value
    return p


def _movie_provider(return_value=None, side_effect=None) -> MovieMetadataProvider:
    p = MagicMock(spec=MovieMetadataProvider)
    if side_effect is not None:
        p.fetch.side_effect = side_effect
    else:
        p.fetch.return_value = return_value
    return p


def _make_event(
    source_type: str = "instagram",
    start_time: datetime | None = ...,
    title: str | None = "Test Event",
    event_id: str = "evt-1",
) -> Event:
    if start_time is ...:
        start_time = datetime(2025, 6, 21, 20, 0, tzinfo=LOCAL_TZ)
    return Event(
        event_id=event_id,
        source_event_candidates=[],
        source_type=source_type,
        created_at=NOW,
        updated_at=NOW,
        title=title,
        start_time=start_time,
    )


def _make_service(
    db_path: Path,
    cfg: AppConfig,
    *,
    weather: WeatherProvider | None = None,
    air_quality: AirQualityProvider | None = None,
    movie: MovieMetadataProvider | None = None,
    rules: list[SyntheticActivityRule] | None = None,
    logger: StructuredLogger | None = None,
) -> EnrichmentService:
    return EnrichmentService(
        weather_provider=weather or _weather_provider(),
        air_quality_provider=air_quality,
        movie_provider=movie,
        astronomical_calculator=AstronomicalCalculator(),
        synthetic_rules=rules or [],
        config=cfg,
        db_path=db_path,
        get_now=lambda: NOW,
        logger=logger,
    )


# ---------------------------------------------------------------------------
# Event with start_time → weather and astro populated
# ---------------------------------------------------------------------------


def _enriched_weather(db_path, cfg, *, start_time=..., day=CLEAR_DAY) -> dict:
    svc = _make_service(db_path, cfg, weather=_weather_provider(day))
    event = _make_event(start_time=start_time)
    return svc.enrich([event], RUN_DATE)[0].weather


def test_event_with_start_time_gets_weather(db_path, cfg):
    weather = _enriched_weather(db_path, cfg)
    assert weather is not None
    assert set(weather) == {"sampled_hour", "forecast", "observed"}


def test_weather_is_sampled_at_the_events_own_hour(db_path, cfg):
    """The daily high says nothing about a 9pm show; hour 20 must score as hour 20."""
    weather = _enriched_weather(db_path, cfg)
    assert weather["sampled_hour"] == 20
    assert weather["forecast"]["hour"]["temperature_f"] == pytest.approx(80.0)


@pytest.mark.parametrize(
    "default_hour,expect_synthetic",
    [(20, 1), (18, 0)],  # CLEAR_DAY runs 60F + hour, so 20 is 80F and 18 is 78F
)
def test_synthetic_activities_judged_at_the_default_hour(
    db_path, cfg, default_hour, expect_synthetic
):
    """Synthetic rules have no start time of their own, so they take the default hour."""
    cfg.weather.default_hour = default_hour
    svc = _make_service(
        db_path, cfg, weather=_weather_provider(CLEAR_DAY), rules=[_walk_rule(min_temp_f=79)]
    )
    results = svc.enrich([], RUN_DATE)
    assert len([e for e in results if e.source_type == "synthetic"]) == expect_synthetic


def test_full_day_series_is_denormalised_onto_the_event(db_path, cfg):
    """A future cache purge must not orphan an event from its own conditions."""
    weather = _enriched_weather(db_path, cfg)
    assert len(weather["forecast"]["day_series"]) == 24


def test_raw_readings_are_kept_not_just_a_derived_score(db_path, cfg):
    """Curves will be retuned; history has to be rescorable without a refetch."""
    hour = _enriched_weather(db_path, cfg)["forecast"]["hour"]
    assert hour["dew_point_f"] == pytest.approx(50.0)
    assert hour["relative_humidity"] == pytest.approx(40.0)
    assert hour["wind_speed_mph"] == pytest.approx(5.0)


def test_forecast_records_when_it_was_issued(db_path, cfg):
    """A forecast's age is what separates it from what actually happened."""
    assert _enriched_weather(db_path, cfg)["forecast"]["issued_at"] == NOW.isoformat()


def test_observed_is_reserved_and_left_empty(db_path, cfg):
    """No backfill job exists yet; the slot exists so adding one is not a migration."""
    assert _enriched_weather(db_path, cfg)["observed"] is None


def test_event_gets_no_weather_when_its_hour_is_missing(db_path, cfg):
    sparse = {"date": "2025-06-21", "hours": [_hour_record(3)]}
    assert _enriched_weather(db_path, cfg, day=sparse) is None


def test_event_with_start_time_gets_astronomical_data(db_path, cfg):
    svc = _make_service(db_path, cfg)
    event = _make_event()
    results = svc.enrich([event], RUN_DATE)
    astro = results[0].astronomical_data
    assert astro is not None
    assert "sunrise" in astro
    assert "sunset" in astro
    assert "dawn" in astro
    assert "dusk" in astro


# ---------------------------------------------------------------------------
# Event with start_time=None → no weather, no astro
# ---------------------------------------------------------------------------


def test_event_with_no_start_time_gets_none_weather(db_path, cfg):
    svc = _make_service(db_path, cfg)
    event = _make_event(start_time=None)
    results = svc.enrich([event], RUN_DATE)
    assert results[0].weather is None


def test_event_with_no_start_time_gets_none_astro(db_path, cfg):
    svc = _make_service(db_path, cfg)
    event = _make_event(start_time=None)
    results = svc.enrich([event], RUN_DATE)
    assert results[0].astronomical_data is None


def test_event_with_no_start_time_does_not_raise(db_path, cfg):
    svc = _make_service(db_path, cfg)
    svc.enrich([_make_event(start_time=None)], RUN_DATE)  # should not raise


# ---------------------------------------------------------------------------
# Provider returns None (e.g. >16 days ahead)
# ---------------------------------------------------------------------------


def test_provider_returns_none_event_weather_is_none(db_path, cfg):
    svc = _make_service(db_path, cfg, weather=_weather_provider(return_value=None))
    event = _make_event()
    results = svc.enrich([event], RUN_DATE)
    assert results[0].weather is None


# ---------------------------------------------------------------------------
# Weather cache: same date → provider called once
# ---------------------------------------------------------------------------


def test_weather_cache_hit_provider_not_called_twice(db_path, cfg):
    wp = _weather_provider(CLEAR_DAY)
    svc = _make_service(db_path, cfg, weather=wp)
    e1 = _make_event(event_id="evt-1")
    e2 = _make_event(event_id="evt-2")  # same date as e1
    svc.enrich([e1, e2], RUN_DATE)
    assert wp.fetch.call_count == 1


def test_weather_cached_in_db_between_calls(db_path, cfg):
    """Second enrich() call re-uses DB cache, provider not called."""
    wp = _weather_provider(CLEAR_DAY)
    svc = _make_service(db_path, cfg, weather=wp)
    svc.enrich([_make_event()], RUN_DATE)
    first_call_count = wp.fetch.call_count

    svc.enrich([_make_event()], RUN_DATE)
    assert wp.fetch.call_count == first_call_count  # no additional fetch


# ---------------------------------------------------------------------------
# Weather provider raises → event retained with weather=None
# ---------------------------------------------------------------------------


def test_weather_provider_raises_event_retained(db_path, cfg):
    svc = _make_service(db_path, cfg, weather=_weather_provider(side_effect=RuntimeError("boom")))
    event = _make_event()
    results = svc.enrich([event], RUN_DATE)
    assert len(results) >= 1
    assert results[0].weather is None


def test_weather_provider_raises_next_event_still_processed(db_path, cfg):
    wp = _weather_provider(side_effect=RuntimeError("boom"))
    svc = _make_service(db_path, cfg, weather=wp)
    e1 = _make_event(event_id="evt-1")
    e2 = _make_event(event_id="evt-2", start_time=datetime(2025, 6, 22, 20, 0, tzinfo=LOCAL_TZ))
    results = svc.enrich([e1, e2], RUN_DATE)
    assert len(results) == 2  # both retained despite failures


def test_weather_provider_raises_error_logged(db_path, cfg):
    logger = MagicMock(spec=StructuredLogger)
    svc = _make_service(
        db_path, cfg,
        weather=_weather_provider(side_effect=RuntimeError("boom")),
        logger=logger,
    )
    svc.enrich([_make_event()], RUN_DATE)
    logger.error.assert_called()


# ---------------------------------------------------------------------------
# Movie provider raises → event retained, error logged
# ---------------------------------------------------------------------------


def test_movie_provider_raises_event_retained(db_path, cfg):
    mp = _movie_provider(side_effect=RuntimeError("TMDb down"))
    svc = _make_service(db_path, cfg, movie=mp)
    event = _make_event(source_type="cinema_veezi")
    results = svc.enrich([event], RUN_DATE)
    assert len(results) >= 1
    assert results[0].metadata == {}


def test_movie_provider_raises_pipeline_continues(db_path, cfg):
    """Exception in movie enrichment does not prevent subsequent events from being processed."""
    mp = _movie_provider(side_effect=RuntimeError("TMDb down"))
    svc = _make_service(db_path, cfg, movie=mp)
    e1 = _make_event(source_type="cinema_veezi", event_id="evt-1")
    e2 = _make_event(source_type="cinema_veezi", event_id="evt-2")
    results = svc.enrich([e1, e2], RUN_DATE)
    assert len(results) == 2


# ---------------------------------------------------------------------------
# Synthetic rules
# ---------------------------------------------------------------------------


def _walk_rule(min_temp_f: float | None = None) -> SyntheticActivityRule:
    return SyntheticActivityRule(
        name="Evening walk",
        conditions=SyntheticConditions(
            min_temp_f=min_temp_f,
            weather=["clear"],
        ),
        tags=["outdoor", "walking"],
        summary="A nice walk",
    )


def test_synthetic_rule_conditions_met_event_appended(db_path, cfg):
    svc = _make_service(
        db_path, cfg,
        weather=_weather_provider(CLEAR_DAY),
        rules=[_walk_rule()],
    )
    results = svc.enrich([_make_event()], RUN_DATE)
    synthetic = [e for e in results if e.source_type == "synthetic"]
    assert len(synthetic) == 1


def test_synthetic_rule_conditions_not_met_no_synthetic_event(db_path, cfg):
    rainy = {"date": "2025-06-21", "hours": [_hour_record(h, "rain") for h in range(24)]}
    svc = _make_service(
        db_path, cfg,
        weather=_weather_provider(rainy),
        rules=[_walk_rule()],
    )
    results = svc.enrich([_make_event()], RUN_DATE)
    synthetic = [e for e in results if e.source_type == "synthetic"]
    assert len(synthetic) == 0


# ---------------------------------------------------------------------------
# Return order: real events first, synthetic last
# ---------------------------------------------------------------------------


def test_returned_list_order_real_first_synthetic_last(db_path, cfg):
    svc = _make_service(
        db_path, cfg,
        weather=_weather_provider(CLEAR_DAY),
        rules=[_walk_rule()],
    )
    real_event = _make_event()
    results = svc.enrich([real_event], RUN_DATE)
    assert len(results) == 2
    assert results[0].source_type != "synthetic"
    assert results[-1].source_type == "synthetic"


def test_no_real_events_synthetic_still_generated(db_path, cfg):
    svc = _make_service(
        db_path, cfg,
        weather=_weather_provider(CLEAR_DAY),
        rules=[_walk_rule()],
    )
    results = svc.enrich([], RUN_DATE)
    assert len(results) == 1
    assert results[0].source_type == "synthetic"


# ---------------------------------------------------------------------------
# Air quality
# ---------------------------------------------------------------------------


AQI_DAY = {"date": "2025-06-21", "hours": [{"hour": h, "us_aqi": 20.0 + h} for h in range(24)]}


def _aqi_provider(return_value=AQI_DAY, side_effect=None):
    p = MagicMock(spec=AirQualityProvider)
    if side_effect is not None:
        p.fetch.side_effect = side_effect
    else:
        p.fetch.return_value = return_value
    return p


def test_air_quality_merged_into_the_sampled_hour(db_path, cfg):
    svc = _make_service(db_path, cfg, air_quality=_aqi_provider())
    weather = svc.enrich([_make_event()], RUN_DATE)[0].weather
    assert weather["forecast"]["hour"]["us_aqi"] == pytest.approx(40.0)


def test_air_quality_skipped_when_disabled(db_path, cfg):
    cfg.weather.air_quality_enabled = False
    aqi = _aqi_provider()
    svc = _make_service(db_path, cfg, air_quality=aqi)
    weather = svc.enrich([_make_event()], RUN_DATE)[0].weather
    assert aqi.fetch.call_count == 0
    assert "us_aqi" not in weather["forecast"]["hour"]


def test_missing_air_quality_leaves_the_rest_of_the_weather_intact(db_path, cfg):
    """Beyond the ~7-day AQI horizon the reading is absent, not bad."""
    svc = _make_service(db_path, cfg, air_quality=_aqi_provider(return_value=None))
    hour = svc.enrich([_make_event()], RUN_DATE)[0].weather["forecast"]["hour"]
    assert "us_aqi" not in hour
    assert hour["temperature_f"] == pytest.approx(80.0)


def test_air_quality_failure_does_not_break_enrichment(db_path, cfg):
    svc = _make_service(db_path, cfg, air_quality=_aqi_provider(side_effect=RuntimeError("boom")))
    results = svc.enrich([_make_event()], RUN_DATE)
    assert results[0].weather is not None


def test_no_air_quality_provider_configured_is_fine(db_path, cfg):
    svc = _make_service(db_path, cfg)
    assert svc.enrich([_make_event()], RUN_DATE)[0].weather is not None
