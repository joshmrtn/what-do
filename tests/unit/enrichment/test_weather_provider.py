"""Unit tests for the Open-Meteo weather provider going through the policy.

The forecast cache moves here from `EnrichmentService`. It is the same object
cached one layer lower: the service stored exactly what the provider returned,
keyed on exactly what the provider was asked for. Putting it at the call means
`RequestPolicy.call` is the one place a stored answer short-circuits a request,
so no caller can reach the network by a path the throttle and the retry do not
see.

What stays with the service is the **bound on what is asked** — only it knows
the provider's horizon, and the adapter making a request cheap is the wrong fix
for one that can never be answered.
"""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone

import pytest
import requests

from src.enrichment.weather import OPEN_METEO_HOST, OpenMeteoProvider, WeatherDayCache
from src.storage.memory.weather_cache import InMemoryWeatherCache
from tests.support.network import fetcher_policy

NOW = datetime(2026, 8, 17, 12, 0, tzinfo=timezone.utc)
DAY = date(2026, 8, 20)
LAT, LNG = 42.52, -70.89
URL = f"https://{OPEN_METEO_HOST}/v1/forecast"


def _hourly(hours: int = 24) -> dict:
    return {
        "time": [f"2026-08-20T{h:02d}:00" for h in range(hours)],
        "temperature_2m": [60.0 + h for h in range(hours)],
        "relative_humidity_2m": [40.0] * hours,
        "dew_point_2m": [50.0] * hours,
        "precipitation": [0.0] * hours,
        "wind_speed_10m": [5.0] * hours,
        "weather_code": [0] * hours,
    }


def _response(status: int = 200, payload: dict | None = None) -> requests.Response:
    response = requests.Response()
    response.status_code = status
    body = payload if payload is not None else {"hourly": _hourly()}
    response._content = json.dumps(body).encode()
    return response


class _FakeSession:
    """Records requests and replays prepared responses."""

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
    sleeps: list[float] | None = None,
) -> OpenMeteoProvider:
    return OpenMeteoProvider(
        session=session,
        policy=fetcher_policy(urls=URL, sleeps=sleeps, now=now),
        weather_cache=cache if cache is not None else InMemoryWeatherCache(),
        cache_ttl=cache_ttl,
        get_now=lambda: now,
    )


# ---------------------------------------------------------------------------
# The fetch itself
# ---------------------------------------------------------------------------


def test_a_day_of_hourly_readings_comes_back():
    day = _provider(_FakeSession(_response())).fetch(DAY, LAT, LNG)
    assert day is not None
    assert day["date"] == "2026-08-20"
    assert len(day["hours"]) == 24


def test_the_timeout_comes_from_the_policy_not_a_hardcoded_number():
    """It was 10 seconds in the module, answerable to nothing."""
    session = _FakeSession(_response())
    _provider(session).fetch(DAY, LAT, LNG)
    assert session.calls[0]["timeout"] == pytest.approx(30.0)


def test_an_empty_series_is_no_day():
    session = _FakeSession(_response(payload={"hourly": _hourly(hours=0)}))
    assert _provider(session).fetch(DAY, LAT, LNG) is None


def test_a_failure_is_still_a_None_rather_than_an_exception():
    """Enrichment treats a missing forecast as normal; it must not raise into
    the batch after the policy has given up."""
    session = _FakeSession(*[_response(status=500) for _ in range(3)])
    assert _provider(session).fetch(DAY, LAT, LNG) is None


# ---------------------------------------------------------------------------
# Retry — the thing `except Exception: return None` used to make impossible
# ---------------------------------------------------------------------------


def test_a_transient_failure_is_retried_rather_than_swallowed():
    """The provider caught every exception and returned None, so a retry had
    nothing to act on. Catching now happens outside the policy, not inside."""
    session = _FakeSession(_response(status=503), _response())
    day = _provider(session).fetch(DAY, LAT, LNG)

    assert day is not None
    assert len(session.calls) == 2


def test_a_bad_request_is_not_retried():
    session = _FakeSession(_response(status=400))
    assert _provider(session).fetch(DAY, LAT, LNG) is None
    assert len(session.calls) == 1


# ---------------------------------------------------------------------------
# The cache, now bound to the call
# ---------------------------------------------------------------------------


def test_a_second_fetch_of_one_day_is_served_from_the_cache():
    store = InMemoryWeatherCache()
    session = _FakeSession(_response())
    _provider(session, cache=store).fetch(DAY, LAT, LNG)
    _provider(session, cache=store).fetch(DAY, LAT, LNG)

    assert len(session.calls) == 1


def test_a_different_day_is_a_different_key():
    store = InMemoryWeatherCache()
    session = _FakeSession(_response(), _response())
    _provider(session, cache=store).fetch(DAY, LAT, LNG)
    _provider(session, cache=store).fetch(date(2026, 8, 21), LAT, LNG)

    assert len(session.calls) == 2


def test_a_forecast_past_its_lifetime_is_refetched():
    """An event found a week out must not score on the forecast issued the day
    it was discovered."""
    store = InMemoryWeatherCache()
    session = _FakeSession(_response(), _response())
    _provider(session, cache=store, now=NOW).fetch(DAY, LAT, LNG)

    later = NOW + timedelta(hours=13)
    _provider(session, cache=store, now=later).fetch(DAY, LAT, LNG)

    assert len(session.calls) == 2


def test_a_declared_never_caches_nothing():
    store = InMemoryWeatherCache()
    session = _FakeSession(_response(), _response())
    _provider(session, cache=store, cache_ttl=None).fetch(DAY, LAT, LNG)
    _provider(session, cache=store, cache_ttl=None).fetch(DAY, LAT, LNG)

    assert len(session.calls) == 2


def test_a_failed_fetch_is_not_stored_as_a_forecast():
    """Caching a miss here would serve "no weather" for the whole lifetime."""
    store = InMemoryWeatherCache()
    _provider(_FakeSession(_response(status=400)), cache=store).fetch(DAY, LAT, LNG)

    session = _FakeSession(_response())
    assert _provider(session, cache=store).fetch(DAY, LAT, LNG) is not None


# ---------------------------------------------------------------------------
# The cache strategy on its own
# ---------------------------------------------------------------------------


def test_the_strategy_stamps_from_the_injected_clock():
    store = InMemoryWeatherCache()
    strategy = WeatherDayCache(
        store,
        day=DAY,
        latitude=LAT,
        longitude=LNG,
        ttl=timedelta(hours=12),
        get_now=lambda: NOW,
    )
    strategy.put({"date": "2026-08-20", "hours": []})

    assert store.get(day=DAY, latitude=LAT, longitude=LNG, fresh_since=NOW) is not None
