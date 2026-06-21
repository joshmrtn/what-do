"""GeocoderProvider abstraction — one implementation per geocoding API."""

from __future__ import annotations

from abc import ABC, abstractmethod

import requests


class GeocoderProvider(ABC):
    """Abstract base for forward geocoding providers."""

    @abstractmethod
    def geocode(self, address: str) -> tuple[float, float] | None:
        """Convert an address string to (latitude, longitude).

        Args:
            address: Human-readable address.

        Returns:
            (lat, lng) tuple, or None if the address could not be resolved.
        """


class NominatimGeocoder(GeocoderProvider):
    """Forward geocoder backed by Nominatim (OpenStreetMap). No API key required."""

    def __init__(self, user_agent: str = "what-do/0.1") -> None:
        self._user_agent = user_agent

    def geocode(self, address: str) -> tuple[float, float] | None:
        """Resolve an address via the Nominatim search endpoint."""
        resp = requests.get(
            "https://nominatim.openstreetmap.org/search",
            params={"q": address, "format": "json", "limit": 1},
            headers={"User-Agent": self._user_agent},
            timeout=10,
        )
        resp.raise_for_status()
        results = resp.json()
        if not results:
            return None
        return float(results[0]["lat"]), float(results[0]["lon"])
