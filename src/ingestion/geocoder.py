"""GeocoderProvider abstraction — one implementation per geocoding API."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod

from src.network.http import HttpFetcher


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


#: Nominatim publishes a **hard limit of one request per second** and blocks
#: clients that break it. It is the one host in this codebase whose allowance is
#: stated rather than guessed at.
NOMINATIM_HOST = "nominatim.openstreetmap.org"


class NominatimGeocoder(GeocoderProvider):
    """Forward geocoder backed by Nominatim (OpenStreetMap). No API key required.

    **Dormant, and the throttle is why that matters.** `_resolve_seed_venues` is
    the only caller and `data/seeds.yaml` holds no venues, so this has never
    made a request. The loop it sits in geocodes once per venue with no spacing
    of its own, so the first populated seeds file would have fired straight into
    a published limit. Going through the policy is what makes that safe before
    it happens rather than after.
    """

    def __init__(self, fetcher: HttpFetcher) -> None:
        """
        Args:
            fetcher: The polite conditional GET. Carries the 1.5s spacing
                declared for this host — deliberately outside Nominatim's own
                1/sec, because being a good neighbour costs nothing here.
        """
        self._fetcher = fetcher

    def geocode(self, address: str) -> tuple[float, float] | None:
        """Resolve an address via the Nominatim search endpoint.

        Returns None when the address could not be resolved, which is an answer
        rather than a failure. A transport failure raises, so the caller sees it.
        """
        body = self._fetcher.get(
            f"https://{NOMINATIM_HOST}/search",
            label="nominatim",
            params={"q": address, "format": "json", "limit": "1"},
        )
        results = json.loads(body)
        if not results:
            return None
        return float(results[0]["lat"]), float(results[0]["lon"])
