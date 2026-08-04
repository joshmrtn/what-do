"""Unit tests for the Open-Meteo air quality provider."""

from datetime import date
from unittest.mock import MagicMock

from src.enrichment.air_quality import AirQualityProvider, OpenMeteoAirQualityProvider

_FETCH_DATE = date(2025, 6, 21)
_LAT, _LNG = 42.52, -70.89
_HOURS = list(range(24))


def _payload() -> dict:
    return {
        "hourly": {
            "time": [f"2025-06-21T{h:02d}:00" for h in _HOURS],
            "us_aqi": [20.0 + h for h in _HOURS],
        }
    }


def _mock_session(payload: dict | None = None, status_error: bool = False):
    resp = MagicMock()
    resp.json.return_value = payload if payload is not None else _payload()
    resp.raise_for_status.side_effect = Exception("400") if status_error else None
    session = MagicMock()
    session.get.return_value = resp
    return session


def _fetch(session) -> dict | None:
    return OpenMeteoAirQualityProvider(session=session).fetch(_FETCH_DATE, _LAT, _LNG)


def test_is_an_air_quality_provider():
    assert isinstance(OpenMeteoAirQualityProvider(session=MagicMock()), AirQualityProvider)


def test_request_asks_for_us_aqi_on_the_target_day():
    session = _mock_session()
    _fetch(session)
    params = session.get.call_args.kwargs["params"]
    assert "us_aqi" in params["hourly"]
    assert params["start_date"] == "2025-06-21"
    assert params["end_date"] == "2025-06-21"
    assert params["timezone"] == "auto"


def test_returns_one_record_per_hour():
    day = _fetch(_mock_session())
    assert [record["hour"] for record in day["hours"]] == _HOURS


def test_reading_is_taken_from_its_own_hour():
    day = _fetch(_mock_session())
    assert day["hours"][20]["us_aqi"] == 40.0


def test_returns_none_on_network_error():
    session = MagicMock()
    session.get.side_effect = Exception("network error")
    assert _fetch(session) is None


def test_returns_none_on_http_error():
    assert _fetch(_mock_session(status_error=True)) is None


def test_returns_none_beyond_the_forecast_horizon():
    """Out-of-range dates return an error payload, not nulls — AQI covers ~7 of 16 days."""
    error_body = {"error": True, "reason": "Parameter 'start_date' is out of allowed range"}
    assert _fetch(_mock_session(error_body)) is None


def test_returns_none_on_empty_series():
    assert _fetch(_mock_session({"hourly": {"time": [], "us_aqi": []}})) is None


def test_missing_reading_for_an_hour_becomes_none():
    payload = _payload()
    payload["hourly"]["us_aqi"][3] = None
    day = _fetch(_mock_session(payload))
    assert day["hours"][3]["us_aqi"] is None
