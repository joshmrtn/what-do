"""VenueSource abstraction — one implementation per geographic data provider."""

from __future__ import annotations

from abc import ABC, abstractmethod

from src.models.venue import Venue


class VenueSource(ABC):
    """Abstract base for venue discovery providers (Overpass, Google Places, etc.)."""

    @abstractmethod
    def fetch_venues(
        self,
        lat: float,
        lng: float,
        radius_miles: float,
        categories: list[str],
    ) -> list[Venue]:
        """Return venues within radius of (lat, lng) matching the given categories.

        Args:
            lat: Centre latitude.
            lng: Centre longitude.
            radius_miles: Search radius in miles.
            categories: Venue category slugs to search for.

        Returns:
            List of discovered Venue objects.
        """
