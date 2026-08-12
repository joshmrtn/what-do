"""Contract every WeatherCacheRepository implementation must satisfy.

`get` takes `fresh_since` rather than returning whatever is stored, so a caller
**cannot** serve a stale forecast by forgetting to check. That is deliberate:
an event found a week out would otherwise score on the forecast issued the day
it was discovered, forever.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import pytest

from src.storage.db import connect, init_db
from src.storage.memory.weather_cache import InMemoryWeatherCache
from src.storage.sqlite.weather_cache import SqliteWeatherCache

_NOW = datetime(2026, 8, 11, 12, 0, tzinfo=timezone.utc)
_DAY = date(2026, 8, 14)
_LAT, _LNG = 42.5, -70.9
_TTL = timedelta(hours=6)


@pytest.fixture(params=["sqlite", "memory"])
def cache(request, tmp_path):
    if request.param == "sqlite":
        path = tmp_path / "weather.db"
        init_db(path)
        return SqliteWeatherCache(path)
    return InMemoryWeatherCache()


def _put(cache, data=None, now=_NOW, day=_DAY, lat=_LAT, lng=_LNG):
    cache.put(day=day, latitude=lat, longitude=lng,
              data=data if data is not None else {"temp_f": 71}, now=now)


def _get(cache, fresh_since=_NOW - _TTL, day=_DAY, lat=_LAT, lng=_LNG):
    return cache.get(day=day, latitude=lat, longitude=lng, fresh_since=fresh_since)


class TestRoundTrip:
    def test_a_stored_forecast_reads_back(self, cache):
        _put(cache, {"temp_f": 71, "hourly": [1, 2, 3]})

        assert _get(cache) == {"temp_f": 71, "hourly": [1, 2, 3]}

    def test_an_empty_cache_returns_nothing(self, cache):
        assert _get(cache) is None

    def test_refetching_the_same_day_replaces_it(self, cache):
        _put(cache, {"temp_f": 60})
        _put(cache, {"temp_f": 80})

        assert _get(cache) == {"temp_f": 80}


class TestKeying:
    def test_another_day_is_a_different_entry(self, cache):
        _put(cache)

        assert _get(cache, day=date(2026, 8, 15)) is None

    def test_another_location_is_a_different_entry(self, cache):
        _put(cache)

        assert _get(cache, lat=40.0) is None
        assert _get(cache, lng=-71.0) is None


class TestFreshness:
    def test_an_entry_older_than_the_window_is_not_served(self, cache):
        _put(cache, now=_NOW - timedelta(hours=9))

        assert _get(cache, fresh_since=_NOW - _TTL) is None

    def test_an_entry_inside_the_window_is_served(self, cache):
        _put(cache, now=_NOW - timedelta(hours=1))

        assert _get(cache, fresh_since=_NOW - _TTL) is not None

    def test_an_entry_exactly_on_the_boundary_is_served(self, cache):
        _put(cache, now=_NOW - _TTL)

        assert _get(cache, fresh_since=_NOW - _TTL) is not None

    def test_a_stale_entry_is_replaced_rather_than_accumulating(self, cache):
        _put(cache, {"temp_f": 60}, now=_NOW - timedelta(hours=9))
        _put(cache, {"temp_f": 80}, now=_NOW)

        assert _get(cache, fresh_since=_NOW - _TTL) == {"temp_f": 80}


class TestCorruptStamp:
    """SQLite-only: the in-memory store holds a real datetime and cannot have one."""

    def test_an_unparseable_stamp_is_treated_as_expired(self, tmp_path):
        """Refetching costs one request; trusting it could serve any age."""
        path = tmp_path / "weather.db"
        init_db(path)
        cache = SqliteWeatherCache(path)
        cache.put(day=_DAY, latitude=_LAT, longitude=_LNG, data={"temp_f": 71}, now=_NOW)

        conn = connect(path)
        try:
            conn.execute("UPDATE weather_cache SET fetched_at = 'not a timestamp'")
            conn.commit()
        finally:
            conn.close()

        assert cache.get(
            day=_DAY, latitude=_LAT, longitude=_LNG, fresh_since=_NOW - _TTL
        ) is None
