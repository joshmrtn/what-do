"""Cinema Salem (Veezi/Vista) showtime adapter."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Callable

import requests

from src.ingestion.candidate_id import derive_candidate_id
from src.ingestion.source import IngestionSource
from src.models.event_candidate import EventCandidate

_VEEZI_BASE = "https://api.us.veezi.com/v1"


class CinemaVeeziAdapter(IngestionSource):
    """Fetches showtimes from Cinema Salem via the Veezi/Vista API."""

    def __init__(
        self,
        api_key: str,
        session: requests.Session | None = None,
        get_now: Callable[[], datetime] = datetime.now,
    ) -> None:
        self._api_key = api_key
        self._session = session or requests.Session()
        self._get_now = get_now

    def fetch(self) -> list[EventCandidate]:
        """Fetch upcoming showtimes from Cinema Salem."""
        response = self._session.get(
            f"{_VEEZI_BASE}/session",
            headers={"VeeziAccessToken": self._api_key},
        )
        response.raise_for_status()
        sessions: list[dict[str, Any]] = response.json()
        return [self._to_candidate(s) for s in sessions]

    def _to_candidate(self, session: dict[str, Any]) -> EventCandidate:
        raw_dt = session.get("ShowDateTime")
        start = (
            datetime.fromisoformat(raw_dt).replace(tzinfo=timezone.utc)
            if raw_dt
            else None
        )
        return EventCandidate(
            id=self._derive_id(session),
            source="cinema_veezi",
            source_type="cinema_veezi",
            title=session.get("FilmTitle"),
            description=session.get("SynopsisShort"),
            venue=session.get("CinemaName"),
            image_url=session.get("PosterUrl"),
            start_time=start,
            raw_published_at=None,
            discovered_at=self._get_now(),
        )

    def _derive_id(self, session: dict[str, Any]) -> str:
        """Build a stable id so a nightly refetch updates the session's row.

        `ScheduledFilmId` identifies the film, not the screening, so the showtime
        is always part of the material.
        """
        film_id = session.get("ScheduledFilmId")
        if film_id:
            return derive_candidate_id("cinema_veezi", film_id, session.get("ShowDateTime"))
        return derive_candidate_id(
            "cinema_veezi",
            session.get("FilmTitle"),
            session.get("ShowDateTime"),
            session.get("CinemaName"),
        )
