"""Weather provider ABC, OpenMeteo implementation, WMO code mapper, and hour sampling."""

from abc import ABC, abstractmethod
from datetime import date, datetime, timedelta
from typing import Any, Callable, Iterable

import requests

from src.config import ConfigError
from src.enrichment.day_cache import DayReadingsCache
from src.network.http import requests_transient_check
from src.network.policy import RequestPolicy
from src.storage.protocols import DayCache

WMO_TO_CONDITION: dict[int, str] = {
    0: "clear",
    1: "clear",
    2: "partly_cloudy",
    3: "overcast",
    45: "overcast",
    48: "overcast",
    51: "rain",
    53: "rain",
    55: "rain",
    56: "rain",
    57: "rain",
    61: "rain",
    63: "rain",
    65: "rain",
    66: "rain",
    67: "rain",
    71: "snow",
    73: "snow",
    75: "snow",
    77: "snow",
    80: "rain",
    81: "rain",
    82: "rain",
    85: "snow",
    86: "snow",
    95: "thunderstorm",
    96: "thunderstorm",
    99: "thunderstorm",
}


def map_wmo_code(code: int) -> str:
    """Map a WMO weather interpretation code to an internal condition string.

    Returns:
        One of: "clear", "partly_cloudy", "overcast", "rain", "snow", "thunderstorm".
        Unknown codes fall back to "overcast".
    """
    return WMO_TO_CONDITION.get(code, "overcast")


#: Open-Meteo hourly variable -> the reading name used everywhere downstream.
#: Names match the comfort curve keys in config, so scoring needs no translation.
HOURLY_VARIABLES: dict[str, str] = {
    "temperature_2m": "temperature_f",
    "relative_humidity_2m": "relative_humidity",
    "dew_point_2m": "dew_point_f",
    "precipitation": "precipitation_mm",
    "wind_speed_10m": "wind_speed_mph",
}


def latest_forecast(events: Iterable[Any]) -> datetime | None:
    """When the freshest forecast behind these events was issued.

    The newest rather than the oldest: a listing is as current as the most
    recent forecast it was scored against, and an event beyond the forecast
    horizon carries an old one forever without making tonight stale.

    Returns:
        The newest `issued_at`, or None when no event carries a forecast — an
        all-indoor listing rather than a failed one.
    """
    issued: list[datetime] = []
    for event in events:
        forecast = (getattr(event, "weather", None) or {}).get("forecast") or {}
        stamp = forecast.get("issued_at")
        if stamp:
            issued.append(datetime.fromisoformat(stamp))
    return max(issued) if issued else None


class WeatherProvider(ABC):
    """Abstract base for weather data providers."""

    @abstractmethod
    def fetch(self, date: date, lat: float, lng: float) -> dict[str, Any] | None:
        """Fetch a full day of hourly weather for a location.

        Returns:
            Dict with keys `date` and `hours` — a list of per-hour records, each
            holding `hour` plus every reading in HOURLY_VARIABLES and `condition`.
            None if the data is unavailable.
        """


def sample_hour(
    day: dict[str, Any], when: datetime | None, default_hour: int
) -> dict[str, Any] | None:
    """Select the hourly record covering `when`.

    Args:
        day: A day as returned by `WeatherProvider.fetch`.
        when: The event's start time. None falls back to `default_hour`, so an
            unknown-time event is judged on a typical evening rather than the
            daily peak.
        default_hour: Local hour to use when `when` is None.

    Returns:
        The matching hourly record, or None if that hour is absent.
    """
    wanted = default_hour if when is None else when.hour
    hours: list[dict[str, Any]] = day.get("hours", [])
    for record in hours:
        if record.get("hour") == wanted:
            return record
    return None


#: The host Open-Meteo's forecast API is reached at, named here so a caller can
#: look up its politeness policy — the cache lifetime among it — without owning
#: a second copy of the address. Air quality is a *separate* host with its own
#: horizon; see `air_quality.py`.
OPEN_METEO_HOST = "api.open-meteo.com"


class OpenMeteoProvider(WeatherProvider):
    """Weather provider backed by the Open-Meteo free API (no key required)."""

    _BASE_URL = f"https://{OPEN_METEO_HOST}/v1/forecast"

    def __init__(
        self,
        *,
        session: requests.Session,
        policy: RequestPolicy,
        weather_cache: DayCache,
        cache_ttl: timedelta | None,
        get_now: Callable[[], datetime],
    ) -> None:
        """
        Args:
            session: Injected HTTP session, so tests never reach the network.
            policy: Throttle, retry and timeout for this host.
            weather_cache: Persistent forecasts, keyed by day and place.
            cache_ttl: Lifetime from the `open_meteo` policy. `None` is a
                declared `never`.
            get_now: Injected clock.
        """
        self._session = session
        self._policy = policy
        self._weather_cache = weather_cache
        self._cache_ttl = cache_ttl
        self._get_now = get_now
        self._is_transient = requests_transient_check(get_now=get_now)

    def fetch(self, date: date, lat: float, lng: float) -> dict[str, Any] | None:
        """Fetch one local day of hourly weather, or serve what is stored.

        Imperial units are requested natively rather than converted, so no
        arithmetic can drift. Humidity and dew point exist only at hourly
        granularity, which is why the daily summary is not enough.

        Returns None on any network, HTTP, or parse error, and on an empty
        series. **The catch sits outside the policy, not inside the request.**
        It used to wrap the call itself, which left retry nothing to act on:
        every failure became a `None` before anything could judge whether it
        was worth another attempt.
        A **narrow** catch. `except Exception` here swallowed a missing import
        as "no readings" — the provider returned None, enrichment recorded an
        absence, and nothing anywhere said a name was undefined. What this
        tolerates is the provider being unreachable, unhappy or malformed;
        what it lets through is us being wrong.
        """
        try:
            return self._policy.call(
                host=OPEN_METEO_HOST,
                perform=lambda timeout: self._request(date, lat, lng, timeout),
                is_transient=self._is_transient,
                cache=DayReadingsCache(
                    self._weather_cache,
                    day=date,
                    latitude=lat,
                    longitude=lng,
                    ttl=self._cache_ttl,
                    get_now=self._get_now,
                ),
                label="weather",
            )
        except ConfigError:
            # `ConfigError` is a `ValueError`, so the narrow catch below would
            # swallow an unassigned host as an absence — and an absence here is
            # indistinguishable from a provider that had nothing to say. That is
            # the swallow this catch was narrowed to prevent, arriving by
            # inheritance instead of by breadth.
            raise
        except (requests.RequestException, ValueError, KeyError):
            return None

    def _request(
        self, day: date, lat: float, lng: float, timeout: float
    ) -> dict[str, Any]:
        """One attempt. Raises so the policy can decide about trying again.

        An empty series raises rather than returning None: the policy caches
        what `perform` returns, and storing "no forecast" would serve that
        absence for the whole lifetime.
        """
        params: dict[str, str | float] = {
            "latitude": lat,
            "longitude": lng,
            "hourly": ",".join([*HOURLY_VARIABLES, "weather_code"]),
            "temperature_unit": "fahrenheit",
            "wind_speed_unit": "mph",
            "precipitation_unit": "mm",
            "timezone": "auto",
            "start_date": day.isoformat(),
            "end_date": day.isoformat(),
        }
        resp = self._session.get(self._BASE_URL, params=params, timeout=timeout)
        resp.raise_for_status()
        hourly = resp.json()["hourly"]
        times = hourly["time"]
        if not times:
            raise ValueError(f"Open-Meteo returned no hours for {day.isoformat()}")
        return {
            "date": day.isoformat(),
            "hours": [self._hour_record(hourly, i) for i in range(len(times))],
        }

    @staticmethod
    def _hour_record(hourly: dict[str, Any], index: int) -> dict[str, Any]:
        """Build one hour's readings. A variable the API omitted becomes None."""

        def value(variable: str) -> float | None:
            series = hourly.get(variable)
            if series is None or index >= len(series) or series[index] is None:
                return None
            return float(series[index])

        record: dict[str, Any] = {
            "hour": datetime.fromisoformat(hourly["time"][index]).hour
        }
        for variable, reading in HOURLY_VARIABLES.items():
            record[reading] = value(variable)

        code = value("weather_code")
        record["condition"] = None if code is None else map_wmo_code(int(code))
        return record
