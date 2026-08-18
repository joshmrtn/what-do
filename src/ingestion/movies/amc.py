"""AMC Showtime API adapter."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Callable

import requests

from src.network.http import requests_transient_check
from src.ingestion.candidate_id import derive_content_id
from src.ingestion.identity import ContentIdRule
from src.network.policy import RequestPolicy
from src.network.protocols import NullCache

from src.ingestion.candidate_id import derive_candidate_id
from src.ingestion.source import IngestionSource
from src.models.event_candidate import EventCandidate
from src.models.source_type import AMC

#: The host AMC's API is reached at, named so a caller can look up its
#: politeness policy without owning a second copy of the address.
AMC_HOST = "api.amctheatres.com"

_AMC_GRAPHQL_URL = f"https://{AMC_HOST}/graphql"

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
        session: requests.Session,
        policy: RequestPolicy,
        get_now: Callable[[], datetime] = datetime.now,
        *,
        uses_content_id: ContentIdRule,
    ) -> None:
        """
        Args:
            api_key: AMC vendor key.
            postal_code: Which theatres to ask about.
            session: Injected HTTP session, so tests never reach the network.
            policy: Throttle, retry and timeout for this host. Used directly
                rather than through `HttpFetcher`, because this is a **POST**
                and a conditional GET cannot express it.
            get_now: Injected clock.
            uses_content_id: Whether AMC's showtime ids may be used as the
                candidate id. Required, so the composition root cannot forget
                to forward it.
        """
        self._api_key = api_key
        self._postal_code = postal_code
        self._session = session
        self._policy = policy
        self._get_now = get_now
        # Required and keyword-only, so the composition root cannot forget it.
        self._uses_content_id = uses_content_id
        self._is_transient = requests_transient_check(get_now=get_now)

    def fetch(self) -> list[EventCandidate]:
        """Fetch upcoming AMC showtimes for the configured postal code."""
        data = self._policy.call(
            host=AMC_HOST,
            perform=self._post,
            is_transient=self._is_transient,
            # A GraphQL POST is not a document with a URL identity, and the
            # showtimes it returns are the thing the pipeline persists. Nothing
            # here is a cacheable resource; the reason is recorded rather than
            # left as an absence somebody has to infer.
            cache=NullCache(
                reason="a GraphQL POST has no URL identity to key on, and its "
                "showtimes are persisted as candidates rather than replayed"
            ),
            label="amc",
        )
        entries = data.get("data", {}).get("getMoviesAndShowtimes", [])
        candidates: list[EventCandidate] = []
        for entry in entries:
            movie = entry.get("movie", {})
            for show in entry.get("showtimes", []):
                candidates.append(self._to_candidate(movie, show))
        return candidates

    def _post(self, timeout: float) -> dict[str, Any]:
        """One attempt. Raises so the policy can decide about trying again."""
        response = self._session.post(
            _AMC_GRAPHQL_URL,
            json={
                "query": _SHOWTIMES_QUERY,
                "variables": {"postalCode": self._postal_code},
            },
            headers={"X-AMC-Vendor-Key": self._api_key},
            timeout=timeout,
        )
        response.raise_for_status()
        payload: dict[str, Any] = response.json()
        return payload

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
            id=self._derive_id(movie, show, start),
            source="amc",
            source_type=AMC,
            title=movie.get("name"),
            description=movie.get("synopsis"),
            image_url=movie.get("posterSrc"),
            venue=show.get("theatre", {}).get("name"),
            start_time=start,
            raw_published_at=None,
            discovered_at=self._get_now(),
        )

    def _derive_id(
        self, movie: dict[str, Any], show: dict[str, Any], start: datetime | None
    ) -> str:
        """Build a stable id so a nightly refetch updates the showtime's row.

        The fallback answers *"AMC published no showtime id"*; the content rule
        answers *"its showtime ids identify nothing"*. Different questions, so
        the fallback keeps its own material and the latched path takes the
        shared listing key.
        """
        if self._uses_content_id("amc"):
            return derive_content_id(
                source="amc",
                title=movie.get("name"),
                venue=show.get("theatre", {}).get("name"),
                start=start,
            )

        showtime_id = show.get("id")
        if showtime_id:
            return derive_candidate_id("amc", showtime_id)
        return derive_candidate_id(
            "amc",
            movie.get("id") or movie.get("name"),
            show.get("showDateTimeUtc"),
            show.get("theatre", {}).get("name"),
        )
