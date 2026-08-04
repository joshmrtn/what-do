"""Comfort curve and weather adjustment tests."""

import pytest

from src.config import ComfortCurve, WeatherConfig
from src.enrichment.comfort import compute_comfort, curve_value


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


TEMP = _curve(ideal=(20.0, 65.0), zero=(-15.0, 78.0), floor=(-40.0, 95.0))


def _config(**overrides: object) -> WeatherConfig:
    """Weather config mirroring config.example.yaml defaults."""
    defaults: dict[str, object] = {
        "max_positive_adjustment": 0.15,
        "max_negative_adjustment": 0.25,
        "comfort": {
            "temperature_f": TEMP,
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
            "snow": -0.4,
            "thunderstorm": -1.0,
            "overcast": -0.2,
            "partly_cloudy": 0.0,
            "clear": 0.0,
        },
    }
    defaults.update(overrides)
    return WeatherConfig(**defaults)  # type: ignore[arg-type]


# --- the curve itself -------------------------------------------------------


@pytest.mark.parametrize("value", [20.0, 42.5, 65.0])
def test_entire_ideal_band_scores_one(value: float) -> None:
    """A plateau, not a peak: band edges score the same as its middle."""
    assert curve_value(TEMP, value) == 1.0


def test_zero_bounds_score_zero() -> None:
    assert curve_value(TEMP, 78.0) == 0.0
    assert curve_value(TEMP, -15.0) == 0.0


def test_floor_bounds_score_minus_one() -> None:
    assert curve_value(TEMP, 95.0) == -1.0
    assert curve_value(TEMP, -40.0) == -1.0


def test_beyond_floor_clamps() -> None:
    assert curve_value(TEMP, 130.0) == -1.0
    assert curve_value(TEMP, -80.0) == -1.0


def test_ramps_interpolate_linearly() -> None:
    # Halfway from ideal_hi (65) to zero_hi (78).
    assert curve_value(TEMP, 71.5) == pytest.approx(0.5)
    # Halfway from zero_hi (78) to floor_hi (95).
    assert curve_value(TEMP, 86.5) == pytest.approx(-0.5)
    # Halfway from ideal_lo (20) to zero_lo (-15).
    assert curve_value(TEMP, 2.5) == pytest.approx(0.5)


def test_sides_ramp_independently() -> None:
    """Cold is tolerated far better than heat, per the configured bounds."""
    assert curve_value(TEMP, -5.0) > 0
    assert curve_value(TEMP, 88.0) < 0


def test_degenerate_side_does_not_divide_by_zero() -> None:
    """Dew point pins all three low bounds together; below them must not blow up."""
    flat_low = _curve((-99.0, 55.0), (-99.0, 65.0), (-99.0, 75.0))
    assert curve_value(flat_low, -120.0) == 1.0


# --- aggregation ------------------------------------------------------------


def test_missing_factor_is_dropped_not_zeroed() -> None:
    """A dropped factor renormalises the remaining weights rather than dragging the mean down."""
    full = compute_comfort(
        {"temperature_f": 45.0, "wind_speed_mph": 5.0, "condition": "clear"}, _config()
    )
    partial = compute_comfort({"temperature_f": 45.0, "condition": "clear"}, _config())
    assert full.comfort == pytest.approx(1.0)
    assert partial.comfort == pytest.approx(1.0)


def test_all_factors_missing_yields_no_adjustment() -> None:
    result = compute_comfort({"condition": "clear"}, _config())
    assert result.comfort == 0.0
    assert result.adjustment == 0.0


def test_humidity_ignored_when_dew_point_present() -> None:
    """Humidity is a fallback for dew point, never a co-factor: they correlate too heavily."""
    with_humidity = compute_comfort(
        {"temperature_f": 45.0, "dew_point_f": 40.0, "relative_humidity": 95.0}, _config()
    )
    without_humidity = compute_comfort(
        {"temperature_f": 45.0, "dew_point_f": 40.0}, _config()
    )
    assert with_humidity.comfort == pytest.approx(without_humidity.comfort)


def test_humidity_scored_when_dew_point_absent() -> None:
    dry = compute_comfort({"temperature_f": 45.0, "relative_humidity": 30.0}, _config())
    muggy = compute_comfort({"temperature_f": 45.0, "relative_humidity": 95.0}, _config())
    assert dry.comfort > muggy.comfort


# --- condition caps ---------------------------------------------------------


def test_thunderstorm_overrides_perfect_numbers() -> None:
    result = compute_comfort(
        {
            "temperature_f": 45.0,
            "dew_point_f": 40.0,
            "wind_speed_mph": 2.0,
            "us_aqi": 10.0,
            "condition": "thunderstorm",
        },
        _config(),
    )
    assert result.comfort == pytest.approx(-1.0)


def test_dry_thunderstorm_is_still_fully_penalised() -> None:
    """Lightning is the hazard, not millimetres — precipitation never supersedes it."""
    result = compute_comfort(
        {
            "temperature_f": 45.0,
            "precipitation_mm": 0.0,
            "condition": "thunderstorm",
        },
        _config(),
    )
    assert result.comfort == pytest.approx(-1.0)


def test_overcast_caps_an_otherwise_perfect_night() -> None:
    result = compute_comfort(
        {"temperature_f": 45.0, "dew_point_f": 40.0, "condition": "overcast"}, _config()
    )
    assert result.comfort == pytest.approx(-0.2)


# --- precipitation supersession --------------------------------------------


def test_light_rain_stays_positive() -> None:
    """A 58F drizzle is a fine night out; the intensity curve supersedes the rain cap."""
    result = compute_comfort(
        {
            "temperature_f": 58.0,
            "dew_point_f": 50.0,
            "precipitation_mm": 0.2,
            "condition": "rain",
        },
        _config(),
    )
    assert result.comfort > 0


def test_heavy_rain_scores_strongly_negative() -> None:
    result = compute_comfort(
        {
            "temperature_f": 58.0,
            "dew_point_f": 50.0,
            "precipitation_mm": 12.0,
            "condition": "rain",
        },
        _config(),
    )
    assert result.comfort < -0.2


def test_rain_falls_back_to_condition_cap_without_a_reading() -> None:
    result = compute_comfort(
        {"temperature_f": 58.0, "dew_point_f": 50.0, "condition": "rain"}, _config()
    )
    assert result.comfort == pytest.approx(-0.4)


def test_rain_is_never_penalised_twice() -> None:
    """With a reading present, the capped result equals the curve-only result."""
    readings = {
        "temperature_f": 58.0,
        "dew_point_f": 50.0,
        "precipitation_mm": 1.0,
    }
    rainy = compute_comfort({**readings, "condition": "rain"}, _config())
    clear = compute_comfort({**readings, "condition": "clear"}, _config())
    assert rainy.comfort == pytest.approx(clear.comfort)


# --- adjustment scaling -----------------------------------------------------


def test_caps_are_asymmetric() -> None:
    """Bad weather vetoes harder than good weather promotes."""
    perfect = compute_comfort(
        {"temperature_f": 45.0, "dew_point_f": 40.0, "condition": "clear"}, _config()
    )
    awful = compute_comfort(
        {"temperature_f": 45.0, "condition": "thunderstorm"}, _config()
    )
    assert perfect.adjustment == pytest.approx(0.15)
    assert awful.adjustment == pytest.approx(-0.25)


def test_adjustment_scales_with_configured_caps() -> None:
    cfg = _config(max_positive_adjustment=1.0, max_negative_adjustment=2.0)
    perfect = compute_comfort({"temperature_f": 45.0, "condition": "clear"}, cfg)
    awful = compute_comfort({"temperature_f": 45.0, "condition": "thunderstorm"}, cfg)
    assert perfect.adjustment == pytest.approx(1.0)
    assert awful.adjustment == pytest.approx(-2.0)


def test_curves_come_from_config_not_code() -> None:
    """Retuning a band changes the verdict with no code change."""
    hot_preferred = _config(
        comfort={"temperature_f": _curve((80.0, 100.0), (60.0, 110.0), (40.0, 120.0))}
    )
    result = compute_comfort({"temperature_f": 90.0, "condition": "clear"}, hot_preferred)
    assert result.comfort == pytest.approx(1.0)


def test_unknown_condition_does_not_cap() -> None:
    result = compute_comfort(
        {"temperature_f": 45.0, "condition": "sandstorm"}, _config()
    )
    assert result.comfort == pytest.approx(1.0)


def test_detail_summarises_the_readings_used() -> None:
    """The ranking engine renders this into a Reason, so it must name real numbers."""
    result = compute_comfort(
        {"temperature_f": 58.0, "dew_point_f": 50.0, "wind_speed_mph": 6.0}, _config()
    )
    assert "58" in result.detail
    assert "50" in result.detail
    assert "6" in result.detail
