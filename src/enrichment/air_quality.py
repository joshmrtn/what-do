"""Air quality provider ABC and the Open-Meteo implementation.

A separate endpoint from the weather forecast, with a much shorter horizon:
about 7 days against the forecast's 16. Readings are therefore absent for most
events, which is why comfort scoring drops a missing factor and renormalises
rather than treating it as poor air.
"""

from abc import ABC, abstractmethod
from datetime import date, datetime
from typing import Any

import requests

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

    _BASE_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"

    def __init__(self, session: requests.Session | None = None) -> None:
        self._session = session or requests.Session()

    def fetch(self, date: date, lat: float, lng: float) -> dict[str, Any] | None:
        """Fetch one local day of hourly air quality.

        Returns None on any network, HTTP, or parse error, on an empty series,
        and on a date outside the forecast horizon — which this endpoint reports
        as an error body rather than null readings.
        """
        params: dict[str, str | float] = {
            "latitude": lat,
            "longitude": lng,
            "hourly": ",".join(AIR_QUALITY_VARIABLES),
            "timezone": "auto",
            "start_date": date.isoformat(),
            "end_date": date.isoformat(),
        }
        try:
            resp = self._session.get(self._BASE_URL, params=params, timeout=10)
            resp.raise_for_status()
            body = resp.json()
            if body.get("error"):
                return None

            hourly = body["hourly"]
            times = hourly["time"]
            if not times:
                return None
            return {
                "date": date.isoformat(),
                "hours": [self._hour_record(hourly, i) for i in range(len(times))],
            }
        except Exception:
            return None

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
