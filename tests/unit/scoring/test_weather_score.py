"""Tests for turning an event's stored weather into a score adjustment."""

from datetime import datetime, timezone
from typing import Any

import pytest

from src.config import ComfortCurve, WeatherConfig
from src.models.event import Event
from src.scoring.weather_score import weather_adjustment

NOW = datetime(2025, 6, 21, 12, 0, tzinfo=timezone.utc)


def _curve(
    ideal: tuple[float, float],
    zero: tuple[float, float],
    floor: tuple[float, float],
    weight: float = 1.0,
    fallback_for: str | None = None,
    supersedes: tuple[str, ...] = (),
) -> ComfortCurve:
    return ComfortCurve(
        ideal=ideal,
        zero=zero,
        floor=floor,
        weight=weight,
        fallback_for=fallback_for,
        supersedes=supersedes,
    )


def _config(**overrides: object) -> WeatherConfig:
    """Weather config mirroring config.example.yaml defaults."""
    defaults: dict[str, object] = {
        "max_positive_adjustment": 0.15,
        "max_negative_adjustment": 0.25,
        "comfort": {
            "temperature_f": _curve((20.0, 65.0), (-15.0, 78.0), (-40.0, 95.0)),
            "dew_point_f": _curve((-99.0, 55.0), (-99.0, 65.0), (-99.0, 75.0)),
            "relative_humidity": _curve(
                (0.0, 45.0), (0.0, 70.0), (0.0, 90.0), fallback_for="dew_point_f"
            ),
            "wind_speed_mph": _curve((0.0, 10.0), (0.0, 20.0), (0.0, 35.0), weight=0.6),
            "us_aqi": _curve((0.0, 50.0), (0.0, 100.0), (0.0, 150.0), weight=0.8),
            "precipitation_mm": _curve(
                (0.0, 0.3), (0.0, 2.5), (0.0, 10.0), weight=0.9, supersedes=("rain", "snow")
            ),
        },
        "condition_penalty": {
            "rain": -0.4,
            "thunderstorm": -1.0,
            "overcast": -0.2,
            "clear": 0.0,
        },
    }
    defaults.update(overrides)
    return WeatherConfig(**defaults)  # type: ignore[arg-type]


PLEASANT = {
    "hour": 20,
    "temperature_f": 62.0,
    "dew_point_f": 50.0,
    "wind_speed_mph": 5.0,
    "precipitation_mm": 0.0,
    "condition": "clear",
}

MISERABLE = {
    "hour": 20,
    "temperature_f": 88.0,
    "dew_point_f": 74.0,
    "wind_speed_mph": 18.0,
    "precipitation_mm": 0.0,
    "condition": "thunderstorm",
}


def _weather(hour: dict[str, Any] | None, observed: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "sampled_hour": 20,
        "forecast": {
            "issued_at": NOW.isoformat(),
            "hour": hour if hour is not None else {},
            "day_series": [],
        },
        "observed": observed,
    }


def _event(setting: str = "outdoor", weather: dict[str, Any] | None = None) -> Event:
    return Event(
        event_id="evt-1",
        source_event_candidates=[],
        source_type="instagram",
        created_at=NOW,
        updated_at=NOW,
        title="Test Event",
        setting=setting,
        weather=weather,
    )


# --- applicability ----------------------------------------------------------


def test_outdoor_event_in_good_weather_is_lifted():
    adjustment, reason = weather_adjustment(_event(weather=_weather(PLEASANT)), _config())

    assert adjustment > 0
    assert reason is not None


def test_outdoor_event_in_bad_weather_is_demoted():
    """Signed, not bonus-only: a miserable night actively pushes an option down."""
    adjustment, reason = weather_adjustment(_event(weather=_weather(MISERABLE)), _config())

    assert adjustment < 0
    assert reason is not None


@pytest.mark.parametrize("setting", ["indoor", "unknown"])
def test_non_outdoor_settings_are_untouched(setting):
    """Perfect weather is irrelevant to an event held inside."""
    adjustment, reason = weather_adjustment(
        _event(setting=setting, weather=_weather(PLEASANT)), _config()
    )

    assert adjustment == 0.0
    assert reason is None


def test_missing_weather_is_not_a_penalty():
    """Beyond the forecast horizon we do not know, so we do not move the score."""
    adjustment, reason = weather_adjustment(_event(weather=None), _config())

    assert adjustment == 0.0
    assert reason is None


def test_empty_hour_record_is_not_a_penalty():
    adjustment, reason = weather_adjustment(_event(weather=_weather(None)), _config())

    assert adjustment == 0.0
    assert reason is None


def test_weather_record_without_a_forecast_does_not_raise():
    event = _event(weather={"sampled_hour": 20, "observed": None})

    assert weather_adjustment(event, _config()) == (0.0, None)


# --- which readings are used ------------------------------------------------


def test_forecast_readings_are_used_when_nothing_was_observed():
    adjustment, _ = weather_adjustment(_event(weather=_weather(MISERABLE)), _config())

    assert adjustment < 0


def test_observed_readings_win_over_the_forecast():
    """A backfilled event must rescore against what happened, not what was predicted."""
    event = _event(weather=_weather(MISERABLE, observed=PLEASANT))

    adjustment, _ = weather_adjustment(event, _config())

    assert adjustment > 0


def test_observed_readings_are_used_even_when_worse():
    event = _event(weather=_weather(PLEASANT, observed=MISERABLE))

    adjustment, _ = weather_adjustment(event, _config())

    assert adjustment < 0


# --- the reason -------------------------------------------------------------


def test_reason_is_attributed_to_the_weather_factor():
    _, reason = weather_adjustment(_event(weather=_weather(PLEASANT)), _config())

    assert reason.factor == "weather_adjustment"


def test_reason_carries_raw_comfort_and_applied_adjustment():
    """similarity holds comfort in -1..+1; contribution holds what was actually applied."""
    adjustment, reason = weather_adjustment(_event(weather=_weather(PLEASANT)), _config())

    assert reason.similarity == pytest.approx(1.0)
    assert reason.contribution == pytest.approx(adjustment)


def test_reason_describes_the_readings_it_used():
    _, reason = weather_adjustment(_event(weather=_weather(PLEASANT)), _config())

    assert "62" in reason.matched_preference
    assert "clear" in reason.matched_preference


def test_reason_direction_follows_the_sign():
    _, good = weather_adjustment(_event(weather=_weather(PLEASANT)), _config())
    _, bad = weather_adjustment(_event(weather=_weather(MISERABLE)), _config())

    assert good.direction == "positive"
    assert bad.direction == "negative"


def test_reason_carries_no_tag():
    """Weather is a property of the night, not of any tag the event was given."""
    _, reason = weather_adjustment(_event(weather=_weather(PLEASANT)), _config())

    assert reason.tag is None


# --- caps -------------------------------------------------------------------


def test_perfect_night_is_capped_at_the_positive_maximum():
    cfg = _config()
    adjustment, _ = weather_adjustment(_event(weather=_weather(PLEASANT)), cfg)

    assert adjustment == pytest.approx(cfg.max_positive_adjustment)


def test_worst_night_is_capped_at_the_negative_maximum():
    cfg = _config()
    adjustment, _ = weather_adjustment(_event(weather=_weather(MISERABLE)), cfg)

    assert adjustment == pytest.approx(-cfg.max_negative_adjustment)


def test_caps_are_read_from_config():
    cfg = _config(max_positive_adjustment=0.5)
    adjustment, _ = weather_adjustment(_event(weather=_weather(PLEASANT)), cfg)

    assert adjustment == pytest.approx(0.5)
