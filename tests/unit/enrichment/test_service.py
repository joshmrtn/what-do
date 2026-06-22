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

CLEAR_WEATHER = {
    "temperature_f": 70.0,
    "condition": "clear",
    "precipitation_mm": 0.0,
    "wind_speed_mph": 5.0,
}


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


def _weather_provider(return_value=CLEAR_WEATHER, side_effect=None) -> WeatherProvider:
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
    movie: MovieMetadataProvider | None = None,
    rules: list[SyntheticActivityRule] | None = None,
    logger: StructuredLogger | None = None,
) -> EnrichmentService:
    return EnrichmentService(
        weather_provider=weather or _weather_provider(),
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


def test_event_with_start_time_gets_weather(db_path, cfg):
    svc = _make_service(db_path, cfg, weather=_weather_provider(CLEAR_WEATHER))
    event = _make_event()
    results = svc.enrich([event], RUN_DATE)
    assert results[0].weather == CLEAR_WEATHER


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
    wp = _weather_provider(CLEAR_WEATHER)
    svc = _make_service(db_path, cfg, weather=wp)
    e1 = _make_event(event_id="evt-1")
    e2 = _make_event(event_id="evt-2")  # same date as e1
    svc.enrich([e1, e2], RUN_DATE)
    assert wp.fetch.call_count == 1


def test_weather_cached_in_db_between_calls(db_path, cfg):
    """Second enrich() call re-uses DB cache, provider not called."""
    wp = _weather_provider(CLEAR_WEATHER)
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
        weather=_weather_provider(CLEAR_WEATHER),
        rules=[_walk_rule()],
    )
    results = svc.enrich([_make_event()], RUN_DATE)
    synthetic = [e for e in results if e.source_type == "synthetic"]
    assert len(synthetic) == 1


def test_synthetic_rule_conditions_not_met_no_synthetic_event(db_path, cfg):
    rainy = {**CLEAR_WEATHER, "condition": "rain"}
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
        weather=_weather_provider(CLEAR_WEATHER),
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
        weather=_weather_provider(CLEAR_WEATHER),
        rules=[_walk_rule()],
    )
    results = svc.enrich([], RUN_DATE)
    assert len(results) == 1
    assert results[0].source_type == "synthetic"
