"""Unit tests for the Open-Meteo air quality provider going through the policy.

This is the caller that had **neither a bound nor a persistent cache**: every
distinct date in a ninety-day listing became a request to a second service, on
every batch run and every read-time rescore, remembered by nothing. The bound
landed in `003a1a4`; the cache lands here.
"""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone

import pytest
import requests

from src.enrichment.air_quality import AIR_QUALITY_HOST, OpenMeteoAirQualityProvider
from src.storage.memory.day_cache import InMemoryDayCache
from tests.support.network import fetcher_policy

NOW = datetime(2026, 8, 17, 12, 0, tzinfo=timezone.utc)
DAY = date(2026, 8, 20)
LAT, LNG = 42.52, -70.89
URL = f"https://{AIR_QUALITY_HOST}/v1/air-quality"


def _payload(hours: int = 24) -> dict:
    return {
        "hourly": {
            "time": [f"2026-08-20T{h:02d}:00" for h in range(hours)],
            "us_aqi": [30 + h for h in range(hours)],
        }
    }


def _response(status: int = 200, payload: dict | None = None) -> requests.Response:
    response = requests.Response()
    response.status_code = status
    response._content = json.dumps(payload if payload is not None else _payload()).encode()
    return response


class _FakeSession:
    def __init__(self, *responses: requests.Response) -> None:
        self._responses = list(responses)
        self.calls: list[dict] = []

    def get(self, url: str, *, params=None, timeout=None):
        self.calls.append({"url": url, "params": params, "timeout": timeout})
        if not self._responses:
            raise AssertionError(f"unexpected extra request to {url}")
        return self._responses.pop(0)


def _provider(
    session,
    *,
    cache=None,
    now: datetime = NOW,
    cache_ttl: timedelta | None = timedelta(hours=12),
) -> OpenMeteoAirQualityProvider:
    return OpenMeteoAirQualityProvider(
        session=session,
        policy=fetcher_policy(urls=URL, now=now),
        air_quality_cache=cache if cache is not None else InMemoryDayCache(),
        cache_ttl=cache_ttl,
        get_now=lambda: now,
    )


def test_a_day_of_readings_comes_back():
    day = _provider(_FakeSession(_response())).fetch(DAY, LAT, LNG)
    assert day is not None
    assert day["date"] == "2026-08-20"
    assert len(day["hours"]) == 24


def test_the_timeout_comes_from_the_policy():
    session = _FakeSession(_response())
    _provider(session).fetch(DAY, LAT, LNG)
    assert session.calls[0]["timeout"] == pytest.approx(30.0)


def test_it_asks_its_own_host_not_the_forecast_one():
    """One provider, two hosts, different horizons. They are not interchangeable."""
    session = _FakeSession(_response())
    _provider(session).fetch(DAY, LAT, LNG)
    assert AIR_QUALITY_HOST in session.calls[0]["url"]


def test_an_error_body_is_no_day():
    """This endpoint reports an out-of-range date as an error body, not nulls."""
    session = _FakeSession(_response(payload={"error": True, "reason": "out of range"}))
    assert _provider(session).fetch(DAY, LAT, LNG) is None


def test_an_empty_series_is_no_day():
    session = _FakeSession(_response(payload=_payload(hours=0)))
    assert _provider(session).fetch(DAY, LAT, LNG) is None


def test_a_failure_is_a_None_rather_than_an_exception():
    """Air quality is advisory: a miss is the normal case and never an error."""
    session = _FakeSession(*[_response(status=500) for _ in range(3)])
    assert _provider(session).fetch(DAY, LAT, LNG) is None


def test_a_transient_failure_is_retried():
    session = _FakeSession(_response(status=503), _response())
    assert _provider(session).fetch(DAY, LAT, LNG) is not None
    assert len(session.calls) == 2


# ---------------------------------------------------------------------------
# The cache this caller has never had
# ---------------------------------------------------------------------------


def test_a_second_fetch_of_one_day_is_served_from_the_cache():
    store = InMemoryDayCache()
    session = _FakeSession(_response())
    _provider(session, cache=store).fetch(DAY, LAT, LNG)
    _provider(session, cache=store).fetch(DAY, LAT, LNG)

    assert len(session.calls) == 1


def test_readings_past_the_lifetime_are_refetched():
    store = InMemoryDayCache()
    session = _FakeSession(_response(), _response())
    _provider(session, cache=store, now=NOW).fetch(DAY, LAT, LNG)
    _provider(session, cache=store, now=NOW + timedelta(hours=13)).fetch(DAY, LAT, LNG)

    assert len(session.calls) == 2


def test_it_does_not_read_the_forecasts_rows():
    """Its own table. Sharing `weather_cache` would collide on
    `UNIQUE (date, latitude, longitude)` and serve a forecast as air quality."""
    forecasts = InMemoryDayCache()
    forecasts.put(
        day=DAY,
        latitude=LAT,
        longitude=LNG,
        data={"date": "2026-08-20", "hours": [{"hour": 20, "temperature_f": 66.0}]},
        now=NOW,
    )

    session = _FakeSession(_response())
    day = _provider(session, cache=InMemoryDayCache()).fetch(DAY, LAT, LNG)

    assert day is not None
    assert "us_aqi" in day["hours"][0]
