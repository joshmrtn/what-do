"""Unit tests for WeatherProvider and WMO code mapping."""

from datetime import date, datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest
import requests

from src.enrichment.weather import (
    OPEN_METEO_HOST,
    OpenMeteoProvider,
    WeatherProvider,
    map_wmo_code,
    sample_hour,
)
from src.config import ConfigError
from src.storage.memory.day_cache import InMemoryDayCache
from tests.support.network import fetcher_policy


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


_FETCH_DATE = date(2025, 6, 21)
_LAT, _LNG = 42.52, -70.89
_NOW = datetime(2025, 6, 21, 12, 0, tzinfo=timezone.utc)


def _provider(session, *, urls: str = f"https://{OPEN_METEO_HOST}/v1/forecast") -> OpenMeteoProvider:
    """The real provider over a faked session.

    Its caching, throttling and retry are the policy's now, so a test about the
    *request* wires the real policy and stands in only for the transport.

    Args:
        session: The faked transport.
        urls: What the policy has a host assignment for. Pointing it somewhere
            else is how a test reaches the unassigned-host path.
    """
    return OpenMeteoProvider(
        session=session,
        policy=fetcher_policy(urls=urls, now=_NOW),
        weather_cache=InMemoryDayCache(),
        cache_ttl=timedelta(hours=12),
        get_now=lambda: _NOW,
    )


#: One reading per hour, so a test can tell which hour was sampled.
_HOURS = list(range(24))


def _hourly_payload(wmo_code: int = 0) -> dict:
    return {
        "hourly": {
            "time": [f"2025-06-21T{h:02d}:00" for h in _HOURS],
            "temperature_2m": [50.0 + h for h in _HOURS],
            "relative_humidity_2m": [40.0 + h for h in _HOURS],
            "dew_point_2m": [30.0 + h for h in _HOURS],
            "precipitation": [0.1 * h for h in _HOURS],
            "wind_speed_10m": [2.0 + h for h in _HOURS],
            "weather_code": [wmo_code for _ in _HOURS],
        }
    }


def _mock_session(payload: dict | None = None):
    mock_resp = MagicMock()
    mock_resp.json.return_value = payload if payload is not None else _hourly_payload()
    mock_resp.raise_for_status.return_value = None
    session = MagicMock()
    session.get.return_value = mock_resp
    return session


def _requested_params(session) -> dict:
    return session.get.call_args.kwargs["params"]


# ---------------------------------------------------------------------------
# OpenMeteoProvider — the hourly request
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "variable",
    [
        "temperature_2m",
        "relative_humidity_2m",
        "dew_point_2m",
        "precipitation",
        "wind_speed_10m",
        "weather_code",
    ],
)
def test_request_asks_for_each_hourly_variable(variable):
    session = _mock_session()
    _provider(session).fetch(_FETCH_DATE, _LAT, _LNG)
    assert variable in _requested_params(session)["hourly"]


def test_request_asks_for_imperial_units():
    """Requested natively so no conversion arithmetic can drift."""
    session = _mock_session()
    _provider(session).fetch(_FETCH_DATE, _LAT, _LNG)
    params = _requested_params(session)
    assert params["temperature_unit"] == "fahrenheit"
    assert params["wind_speed_unit"] == "mph"


def test_request_is_scoped_to_the_single_local_day():
    session = _mock_session()
    _provider(session).fetch(_FETCH_DATE, _LAT, _LNG)
    params = _requested_params(session)
    assert params["start_date"] == "2025-06-21"
    assert params["end_date"] == "2025-06-21"
    assert params["timezone"] == "auto"


# ---------------------------------------------------------------------------
# OpenMeteoProvider — the returned day
# ---------------------------------------------------------------------------


def test_fetch_returns_every_hour_of_the_day():
    day = _provider(_mock_session()).fetch(_FETCH_DATE, _LAT, _LNG)
    assert day is not None
    assert [record["hour"] for record in day["hours"]] == _HOURS


def test_each_hour_carries_every_reading():
    day = _provider(_mock_session()).fetch(_FETCH_DATE, _LAT, _LNG)
    assert set(day["hours"][0]) == {
        "hour",
        "temperature_f",
        "relative_humidity",
        "dew_point_f",
        "precipitation_mm",
        "wind_speed_mph",
        "condition",
    }


def test_readings_are_taken_from_their_own_hour():
    day = _provider(_mock_session()).fetch(_FETCH_DATE, _LAT, _LNG)
    assert day["hours"][20]["temperature_f"] == pytest.approx(70.0)
    assert day["hours"][20]["dew_point_f"] == pytest.approx(50.0)
    assert day["hours"][20]["wind_speed_mph"] == pytest.approx(22.0)


def test_condition_derived_from_the_hourly_wmo_code():
    day = _provider(_mock_session(_hourly_payload(95))).fetch(
        _FETCH_DATE, _LAT, _LNG
    )
    assert day["hours"][0]["condition"] == "thunderstorm"


def test_fetch_records_the_date_it_covers():
    day = _provider(_mock_session()).fetch(_FETCH_DATE, _LAT, _LNG)
    assert day["date"] == "2025-06-21"


# ---------------------------------------------------------------------------
# OpenMeteoProvider — failure modes
# ---------------------------------------------------------------------------


def test_provider_returns_none_on_network_error():
    session = MagicMock()
    session.get.side_effect = requests.ConnectionError("network error")
    assert _provider(session).fetch(_FETCH_DATE, _LAT, _LNG) is None


def test_a_programming_error_is_not_reported_as_no_weather():
    """The catch is narrow on purpose.

    `except Exception` swallowed a missing import as an absent forecast: the
    provider answered None, enrichment recorded no weather, and nothing said a
    name was undefined. A bug must not be able to look like bad weather.
    """
    session = MagicMock()
    session.get.side_effect = NameError("DayReadingsCache")
    with pytest.raises(NameError):
        _provider(session).fetch(_FETCH_DATE, _LAT, _LNG)


def test_provider_returns_none_on_http_error():
    mock_resp = MagicMock()
    mock_resp.raise_for_status.side_effect = requests.HTTPError("500")
    session = MagicMock()
    session.get.return_value = mock_resp
    assert _provider(session).fetch(_FETCH_DATE, _LAT, _LNG) is None


def test_provider_returns_none_on_malformed_payload():
    assert _provider(_mock_session({"unexpected": {}})).fetch(
        _FETCH_DATE, _LAT, _LNG
    ) is None


def test_provider_returns_none_on_empty_series():
    empty = {"hourly": {"time": [], "temperature_2m": []}}
    assert _provider(_mock_session(empty)).fetch(_FETCH_DATE, _LAT, _LNG) is None


def test_provider_tolerates_a_missing_optional_variable():
    """A variable absent from the response becomes None, not a crash."""
    payload = _hourly_payload()
    del payload["hourly"]["dew_point_2m"]
    day = _provider(_mock_session(payload)).fetch(_FETCH_DATE, _LAT, _LNG)
    assert day is not None
    assert day["hours"][0]["dew_point_f"] is None


# ---------------------------------------------------------------------------
# sample_hour
# ---------------------------------------------------------------------------


def _day() -> dict:
    return _provider(_mock_session()).fetch(_FETCH_DATE, _LAT, _LNG)


def test_sample_hour_picks_the_hour_containing_the_start_time():
    record = sample_hour(_day(), datetime(2025, 6, 21, 21, 45), default_hour=20)
    assert record is not None
    assert record["hour"] == 21


def test_sample_hour_without_a_start_time_uses_the_configured_default():
    """An unknown-time event is judged on a typical evening, not the daily peak."""
    record = sample_hour(_day(), None, default_hour=20)
    assert record is not None
    assert record["hour"] == 20


def test_sample_hour_returns_none_when_the_hour_is_absent():
    partial = {"date": "2025-06-21", "hours": [{"hour": 3, "temperature_f": 40.0}]}
    assert sample_hour(partial, datetime(2025, 6, 21, 20, 0), default_hour=20) is None


def test_sample_hour_returns_none_for_an_empty_day():
    assert sample_hour({"date": "2025-06-21", "hours": []}, None, default_hour=20) is None


def test_an_unassigned_host_is_raised_not_reported_as_no_readings():
    """`ConfigError` subclasses `ValueError`, and the catch below names
    `ValueError` — so an unconfigured host would return `None`, enrichment would
    record an absence, and the run would report success having silently scored
    every event with no weather at all.

    That is the failure the narrow catch was written to prevent: what it
    tolerates is the provider being unreachable, unhappy or malformed; what it
    must let through is us being wrong.
    """
    provider = _provider(_mock_session(), urls="https://not-open-meteo.test/x")

    with pytest.raises(ConfigError):
        provider.fetch(_FETCH_DATE, _LAT, _LNG)
