"""AMC Showtime API adapter."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Callable

import requests

from src.ingestion.candidate_id import derive_candidate_id
from src.ingestion.source import IngestionSource
from src.models.event_candidate import EventCandidate

_AMC_GRAPHQL_URL = "https://api.amctheatres.com/graphql"

_SHOWTIMES_QUERY = """
query GetShowtimes($postalCode: String!) {
  getMoviesAndShowtimes(postalCode: $postalCode) {
    movie { name synopsis posterSrc id }
    showtimes { showDateTimeUtc theatre { name } id }
  }
}
"""


class AmcAdapter(IngestionSource):
    """Fetches showtimes from AMC theaters via the AMC Showtime API."""

    def __init__(
        self,
        api_key: str,
        postal_code: str,
        session: requests.Session | None = None,
        get_now: Callable[[], datetime] = datetime.now,
    ) -> None:
        self._api_key = api_key
        self._postal_code = postal_code
        self._session = session or requests.Session()
        self._get_now = get_now

    def fetch(self) -> list[EventCandidate]:
        """Fetch upcoming AMC showtimes for the configured postal code."""
        response = self._session.post(
            _AMC_GRAPHQL_URL,
            json={"query": _SHOWTIMES_QUERY, "variables": {"postalCode": self._postal_code}},
            headers={"X-AMC-Vendor-Key": self._api_key},
        )
        response.raise_for_status()
        data = response.json()
        entries = data.get("data", {}).get("getMoviesAndShowtimes", [])
        candidates: list[EventCandidate] = []
        for entry in entries:
            movie = entry.get("movie", {})
            for show in entry.get("showtimes", []):
                candidates.append(self._to_candidate(movie, show))
        return candidates

    def _to_candidate(
        self, movie: dict[str, Any], show: dict[str, Any]
    ) -> EventCandidate:
        raw_dt = show.get("showDateTimeUtc")
        start = (
            datetime.fromisoformat(raw_dt).replace(tzinfo=timezone.utc)
            if raw_dt
            else None
        )
        return EventCandidate(
            id=self._derive_id(movie, show),
            source="amc",
            source_type="amc",
            title=movie.get("name"),
            description=movie.get("synopsis"),
            image_url=movie.get("posterSrc"),
            venue=show.get("theatre", {}).get("name"),
            start_time=start,
            raw_published_at=None,
            discovered_at=self._get_now(),
        )

    def _derive_id(self, movie: dict[str, Any], show: dict[str, Any]) -> str:
        """Build a stable id so a nightly refetch updates the showtime's row."""
        showtime_id = show.get("id")
        if showtime_id:
            return derive_candidate_id("amc", showtime_id)
        return derive_candidate_id(
            "amc",
            movie.get("id") or movie.get("name"),
            show.get("showDateTimeUtc"),
            show.get("theatre", {}).get("name"),
        )
