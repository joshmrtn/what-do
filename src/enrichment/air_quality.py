"""Air quality provider ABC and the Open-Meteo implementation.

A separate endpoint from the weather forecast, with a much shorter horizon:
about 7 days against the forecast's 16. Readings are therefore absent for most
events, which is why comfort scoring drops a missing factor and renormalises
rather than treating it as poor air.
"""

from abc import ABC, abstractmethod
from datetime import date, datetime, timedelta
from typing import Any, Callable

import requests

from src.enrichment.day_cache import DayReadingsCache
from src.network.http import requests_transient_check
from src.network.policy import RequestPolicy
from src.storage.protocols import DayCache

#: The host this endpoint lives at. Deliberately not the forecast's: it is a
#: separate service with a **shorter horizon**, and one policy covering both
#: hosts still leaves the bound per caller, because they do not share a range.
AIR_QUALITY_HOST = "air-quality-api.open-meteo.com"

#: Air quality readings merged into the hourly weather record.
AIR_QUALITY_VARIABLES: dict[str, str] = {"us_aqi": "us_aqi"}


class AirQualityProvider(ABC):
    """Abstract base for air quality data providers."""

    @abstractmethod
    def fetch(self, date: date, lat: float, lng: float) -> dict[str, Any] | None:
        """Fetch a day of hourly air quality for a location.

        Returns:
            Dict with keys `date` and `hours`, matching the weather provider's
            shape so the same hour sampling works on both. None when unavailable.
        """


class OpenMeteoAirQualityProvider(AirQualityProvider):
    """Air quality provider backed by the free Open-Meteo endpoint (no key required)."""

    _BASE_URL = f"https://{AIR_QUALITY_HOST}/v1/air-quality"

    def __init__(
        self,
        *,
        session: requests.Session,
        policy: RequestPolicy,
        air_quality_cache: DayCache,
        cache_ttl: timedelta | None,
        get_now: Callable[[], datetime],
    ) -> None:
        """
        Args:
            session: Injected HTTP session, so tests never reach the network.
            policy: Throttle, retry and timeout for this host.
            air_quality_cache: Its **own** table. Sharing the forecast's would
                collide on `UNIQUE (date, latitude, longitude)` and serve a
                forecast as air quality.
            cache_ttl: Lifetime from the `open_meteo` policy. `None` is a
                declared `never`.
            get_now: Injected clock.
        """
        self._session = session
        self._policy = policy
        self._cache = air_quality_cache
        self._cache_ttl = cache_ttl
        self._get_now = get_now
        self._is_transient = requests_transient_check(get_now=get_now)

    def fetch(self, date: date, lat: float, lng: float) -> dict[str, Any] | None:
        """Fetch one local day of hourly air quality, or serve what is stored.

        Returns None on any network, HTTP, or parse error, on an empty series,
        and on a date outside the forecast horizon — which this endpoint reports
        as an error body rather than null readings.

        **This caller had neither a bound nor a cache.** The bound is the
        service's, because only it knows this endpoint's horizon is shorter than
        the forecast's. The cache is here, at the request.
        A **narrow** catch. `except Exception` here swallowed a missing import
        as "no readings" — the provider returned None, enrichment recorded an
        absence, and nothing anywhere said a name was undefined. What this
        tolerates is the provider being unreachable, unhappy or malformed;
        what it lets through is us being wrong.
        """
        try:
            return self._policy.call(
                host=AIR_QUALITY_HOST,
                perform=lambda timeout: self._request(date, lat, lng, timeout),
                is_transient=self._is_transient,
                cache=DayReadingsCache(
                    self._cache,
                    day=date,
                    latitude=lat,
                    longitude=lng,
                    ttl=self._cache_ttl,
                    get_now=self._get_now,
                ),
                label="air_quality",
            )
        except (requests.RequestException, ValueError, KeyError):
            return None

    def _request(
        self, day: date, lat: float, lng: float, timeout: float
    ) -> dict[str, Any]:
        """One attempt. Raises so the policy can decide about trying again."""
        params: dict[str, str | float] = {
            "latitude": lat,
            "longitude": lng,
            "hourly": ",".join(AIR_QUALITY_VARIABLES),
            "timezone": "auto",
            "start_date": day.isoformat(),
            "end_date": day.isoformat(),
        }
        resp = self._session.get(self._BASE_URL, params=params, timeout=timeout)
        resp.raise_for_status()
        body = resp.json()
        if body.get("error"):
            raise ValueError(f"Open-Meteo air quality declined {day.isoformat()}")

        hourly = body["hourly"]
        times = hourly["time"]
        if not times:
            raise ValueError(f"Open-Meteo air quality returned no hours for {day}")
        return {
            "date": day.isoformat(),
            "hours": [self._hour_record(hourly, i) for i in range(len(times))],
        }

    @staticmethod
    def _hour_record(hourly: dict[str, Any], index: int) -> dict[str, Any]:
        """Build one hour's readings. A variable the API omitted becomes None."""
        record: dict[str, Any] = {
            "hour": datetime.fromisoformat(hourly["time"][index]).hour
        }
        for variable, reading in AIR_QUALITY_VARIABLES.items():
            series = hourly.get(variable)
            if series is None or index >= len(series) or series[index] is None:
                record[reading] = None
            else:
                record[reading] = float(series[index])
        return record
