"""Unit tests for AstronomicalCalculator."""

from datetime import date, timedelta, timezone

import pytest

from src.enrichment.astronomical import AstronomicalCalculator, AstronomicalData

SALEM_LAT = 42.52
SALEM_LNG = -70.89
SALEM_TZ = "America/New_York"
SOLSTICE = date(2025, 6, 21)


@pytest.fixture
def calc():
    return AstronomicalCalculator()


def test_returns_astronomical_data(calc):
    result = calc.calculate(SOLSTICE, SALEM_LAT, SALEM_LNG, SALEM_TZ)
    assert isinstance(result, AstronomicalData)


def test_all_fields_populated(calc):
    result = calc.calculate(SOLSTICE, SALEM_LAT, SALEM_LNG, SALEM_TZ)
    assert result.sunrise is not None
    assert result.sunset is not None
    assert result.dawn is not None
    assert result.dusk is not None


def test_all_datetimes_timezone_aware(calc):
    result = calc.calculate(SOLSTICE, SALEM_LAT, SALEM_LNG, SALEM_TZ)
    for field in (result.sunrise, result.sunset, result.dawn, result.dusk):
        assert field.tzinfo is not None, f"{field} is not timezone-aware"


def test_sunrise_before_sunset(calc):
    result = calc.calculate(SOLSTICE, SALEM_LAT, SALEM_LNG, SALEM_TZ)
    assert result.sunrise < result.sunset


def test_dawn_before_sunrise(calc):
    result = calc.calculate(SOLSTICE, SALEM_LAT, SALEM_LNG, SALEM_TZ)
    assert result.dawn < result.sunrise


def test_sunset_before_dusk(calc):
    result = calc.calculate(SOLSTICE, SALEM_LAT, SALEM_LNG, SALEM_TZ)
    assert result.sunset < result.dusk


def test_determinism(calc):
    r1 = calc.calculate(SOLSTICE, SALEM_LAT, SALEM_LNG, SALEM_TZ)
    r2 = calc.calculate(SOLSTICE, SALEM_LAT, SALEM_LNG, SALEM_TZ)
    assert r1.sunrise == r2.sunrise
    assert r1.sunset == r2.sunset
    assert r1.dawn == r2.dawn
    assert r1.dusk == r2.dusk


def test_known_sunrise_salem_solstice(calc):
    # Sunrise on 2025-06-21 in Salem MA is ~5:08 AM EDT; assert within ±10 min
    import zoneinfo

    result = calc.calculate(SOLSTICE, SALEM_LAT, SALEM_LNG, SALEM_TZ)
    edt = zoneinfo.ZoneInfo("America/New_York")
    expected_hour, expected_minute = 5, 8
    sr_local = result.sunrise.astimezone(edt)
    delta = abs(
        sr_local.hour * 60 + sr_local.minute - (expected_hour * 60 + expected_minute)
    )
    assert delta <= 10, f"Sunrise {sr_local.strftime('%H:%M')} not within 10 min of 05:08"


def test_known_sunset_salem_solstice(calc):
    # Sunset on 2025-06-21 in Salem MA is ~8:24 PM EDT; assert within ±10 min
    import zoneinfo

    result = calc.calculate(SOLSTICE, SALEM_LAT, SALEM_LNG, SALEM_TZ)
    edt = zoneinfo.ZoneInfo("America/New_York")
    expected_hour, expected_minute = 20, 24
    ss_local = result.sunset.astimezone(edt)
    delta = abs(
        ss_local.hour * 60 + ss_local.minute - (expected_hour * 60 + expected_minute)
    )
    assert delta <= 10, f"Sunset {ss_local.strftime('%H:%M')} not within 10 min of 20:24"


def test_different_dates_give_different_results(calc):
    r1 = calc.calculate(SOLSTICE, SALEM_LAT, SALEM_LNG, SALEM_TZ)
    r2 = calc.calculate(date(2025, 12, 21), SALEM_LAT, SALEM_LNG, SALEM_TZ)
    assert r1.sunrise != r2.sunrise
    assert r1.sunset != r2.sunset
