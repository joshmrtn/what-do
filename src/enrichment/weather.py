"""Weather provider ABC, OpenMeteo implementation, and WMO code mapper."""

from abc import ABC, abstractmethod
from datetime import date

import requests

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


class WeatherProvider(ABC):
    """Abstract base for weather data providers."""

    @abstractmethod
    def fetch(self, date: date, lat: float, lng: float) -> dict | None:
        """Fetch weather for a date and location.

        Returns:
            Dict with keys temperature_f, condition, precipitation_mm, wind_speed_mph,
            or None if the data is unavailable.
        """


class OpenMeteoProvider(WeatherProvider):
    """Weather provider backed by the Open-Meteo free API (no key required)."""

    _BASE_URL = "https://api.open-meteo.com/v1/forecast"

    def __init__(self, session: requests.Session | None = None) -> None:
        self._session = session or requests.Session()

    def fetch(self, date: date, lat: float, lng: float) -> dict | None:
        """Fetch daily weather summary from Open-Meteo.

        Returns temperatures in Fahrenheit and wind speed in mph.
        Returns None on any network, HTTP, or parse error.
        """
        params = {
            "latitude": lat,
            "longitude": lng,
            "daily": "weathercode,temperature_2m_max,precipitation_sum,windspeed_10m_max",
            "temperature_unit": "celsius",
            "windspeed_unit": "kmh",
            "timezone": "auto",
            "start_date": date.isoformat(),
            "end_date": date.isoformat(),
        }
        try:
            resp = self._session.get(self._BASE_URL, params=params, timeout=10)
            resp.raise_for_status()
            daily = resp.json()["daily"]
            temp_c: float = daily["temperature_2m_max"][0]
            precip_mm: float = daily["precipitation_sum"][0]
            wind_kmh: float = daily["windspeed_10m_max"][0]
            wmo_code: int = daily["weathercode"][0]
            return {
                "temperature_f": (temp_c * 9 / 5) + 32,
                "condition": map_wmo_code(wmo_code),
                "precipitation_mm": float(precip_mm),
                "wind_speed_mph": wind_kmh * 0.621371,
            }
        except Exception:
            return None
