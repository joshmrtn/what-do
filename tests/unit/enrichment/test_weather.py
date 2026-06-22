"""Unit tests for WeatherProvider and WMO code mapping."""

from datetime import date
from unittest.mock import MagicMock

import pytest

from src.enrichment.weather import OpenMeteoProvider, WeatherProvider, map_wmo_code


# ---------------------------------------------------------------------------
# WMO code mapping
# ---------------------------------------------------------------------------


def test_wmo_code_0_is_clear():
    assert map_wmo_code(0) == "clear"


def test_wmo_code_1_is_clear():
    assert map_wmo_code(1) == "clear"


def test_wmo_code_2_is_partly_cloudy():
    assert map_wmo_code(2) == "partly_cloudy"


def test_wmo_code_3_is_overcast():
    assert map_wmo_code(3) == "overcast"


@pytest.mark.parametrize("code", [51, 53, 55, 56, 57, 61, 63, 65, 66, 67])
def test_wmo_rain_codes(code):
    assert map_wmo_code(code) == "rain"


@pytest.mark.parametrize("code", [71, 73, 75, 77])
def test_wmo_snow_codes(code):
    assert map_wmo_code(code) == "snow"


@pytest.mark.parametrize("code", [95, 96, 99])
def test_wmo_thunderstorm_codes(code):
    assert map_wmo_code(code) == "thunderstorm"


def test_unknown_wmo_code_falls_back_to_overcast():
    assert map_wmo_code(999) == "overcast"


def test_unmapped_code_falls_back_to_overcast():
    assert map_wmo_code(42) == "overcast"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_session(wmo_code: int, temp_c: float, precip_mm: float, wind_kmh: float):
    mock_resp = MagicMock()
    mock_resp.json.return_value = {
        "daily": {
            "time": ["2025-06-21"],
            "weathercode": [wmo_code],
            "temperature_2m_max": [temp_c],
            "precipitation_sum": [precip_mm],
            "windspeed_10m_max": [wind_kmh],
        }
    }
    mock_resp.raise_for_status.return_value = None
    session = MagicMock()
    session.get.return_value = mock_resp
    return session


_FETCH_DATE = date(2025, 6, 21)
_LAT, _LNG = 42.52, -70.89


# ---------------------------------------------------------------------------
# OpenMeteoProvider — happy path
# ---------------------------------------------------------------------------


def test_provider_returns_dict_with_all_fields():
    provider = OpenMeteoProvider(session=_mock_session(0, 20.0, 0.0, 15.0))
    result = provider.fetch(_FETCH_DATE, _LAT, _LNG)
    assert result is not None
    assert set(result.keys()) == {"temperature_f", "condition", "precipitation_mm", "wind_speed_mph"}


def test_field_types_are_correct():
    provider = OpenMeteoProvider(session=_mock_session(0, 20.0, 1.5, 15.0))
    result = provider.fetch(_FETCH_DATE, _LAT, _LNG)
    assert isinstance(result["temperature_f"], float)
    assert isinstance(result["condition"], str)
    assert isinstance(result["precipitation_mm"], float)
    assert isinstance(result["wind_speed_mph"], float)


def test_temperature_converted_celsius_to_fahrenheit():
    # 0°C → 32°F
    provider = OpenMeteoProvider(session=_mock_session(0, 0.0, 0.0, 0.0))
    result = provider.fetch(_FETCH_DATE, _LAT, _LNG)
    assert result["temperature_f"] == pytest.approx(32.0)


def test_temperature_conversion_non_zero():
    # 20°C → 68°F
    provider = OpenMeteoProvider(session=_mock_session(0, 20.0, 0.0, 0.0))
    result = provider.fetch(_FETCH_DATE, _LAT, _LNG)
    assert result["temperature_f"] == pytest.approx(68.0)


def test_wind_speed_converted_kmh_to_mph():
    # 100 km/h → 62.1371 mph
    provider = OpenMeteoProvider(session=_mock_session(0, 0.0, 0.0, 100.0))
    result = provider.fetch(_FETCH_DATE, _LAT, _LNG)
    assert result["wind_speed_mph"] == pytest.approx(62.1371, rel=1e-3)


def test_condition_derived_from_wmo_code():
    provider = OpenMeteoProvider(session=_mock_session(95, 20.0, 5.0, 30.0))
    result = provider.fetch(_FETCH_DATE, _LAT, _LNG)
    assert result["condition"] == "thunderstorm"


def test_precipitation_passed_through_unchanged():
    provider = OpenMeteoProvider(session=_mock_session(0, 20.0, 3.7, 10.0))
    result = provider.fetch(_FETCH_DATE, _LAT, _LNG)
    assert result["precipitation_mm"] == pytest.approx(3.7)


# ---------------------------------------------------------------------------
# OpenMeteoProvider — failure cases
# ---------------------------------------------------------------------------


def test_provider_returns_none_on_network_error():
    session = MagicMock()
    session.get.side_effect = Exception("network error")
    provider = OpenMeteoProvider(session=session)
    assert provider.fetch(_FETCH_DATE, _LAT, _LNG) is None


def test_provider_returns_none_on_http_error():
    mock_resp = MagicMock()
    mock_resp.raise_for_status.side_effect = Exception("HTTP 500")
    session = MagicMock()
    session.get.return_value = mock_resp
    provider = OpenMeteoProvider(session=session)
    assert provider.fetch(_FETCH_DATE, _LAT, _LNG) is None


def test_provider_returns_none_on_malformed_response():
    mock_resp = MagicMock()
    mock_resp.raise_for_status.return_value = None
    mock_resp.json.return_value = {"unexpected": "structure"}
    session = MagicMock()
    session.get.return_value = mock_resp
    provider = OpenMeteoProvider(session=session)
    assert provider.fetch(_FETCH_DATE, _LAT, _LNG) is None


# ---------------------------------------------------------------------------
# ABC conformance
# ---------------------------------------------------------------------------


def test_weather_provider_is_abstract():
    """WeatherProvider cannot be instantiated directly."""
    with pytest.raises(TypeError):
        WeatherProvider()


def test_open_meteo_is_weather_provider():
    provider = OpenMeteoProvider(session=MagicMock())
    assert isinstance(provider, WeatherProvider)
