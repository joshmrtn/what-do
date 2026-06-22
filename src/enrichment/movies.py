"""Movie metadata provider ABC, TMDb implementation, and event enrichment helper."""

from abc import ABC, abstractmethod

import requests

from src.models.event import Event
from src.utils.logging import StructuredLogger

_MOVIE_SOURCE_TYPES = {"cinema_veezi", "amc"}


class MovieMetadataProvider(ABC):
    """Abstract base for movie metadata providers."""

    @abstractmethod
    def fetch(self, title: str, year: int | None) -> dict | None:
        """Fetch metadata for a movie title.

        Returns:
            Dict with keys genres, runtime_minutes, summary, release_year,
            or None if the movie was not found.
        """


class TMDbProvider(MovieMetadataProvider):
    """Movie metadata provider backed by The Movie Database (TMDb) API."""

    _BASE_URL = "https://api.themoviedb.org/3"

    def __init__(self, api_key: str, session: requests.Session | None = None) -> None:
        self._api_key = api_key
        self._session = session or requests.Session()

    def fetch(self, title: str, year: int | None) -> dict | None:
        """Search TMDb for a movie and return structured metadata.

        Returns:
            Dict with genres, runtime_minutes, summary, release_year,
            or None on not-found or any error.
        """
        try:
            params: dict = {"api_key": self._api_key, "query": title}
            if year is not None:
                params["year"] = year

            search_resp = self._session.get(
                f"{self._BASE_URL}/search/movie", params=params, timeout=10
            )
            search_resp.raise_for_status()
            results = search_resp.json().get("results", [])
            if not results:
                return None

            movie_id = results[0]["id"]
            release_date: str = results[0].get("release_date", "")

            detail_resp = self._session.get(
                f"{self._BASE_URL}/movie/{movie_id}",
                params={"api_key": self._api_key},
                timeout=10,
            )
            detail_resp.raise_for_status()
            detail = detail_resp.json()

            genres = [g["name"] for g in detail.get("genres", [])]
            runtime = detail.get("runtime")
            overview = detail.get("overview") or None
            release_year_str = (detail.get("release_date") or release_date)[:4]
            release_year = int(release_year_str) if release_year_str.isdigit() else None

            return {
                "genres": genres,
                "runtime_minutes": int(runtime) if runtime else None,
                "summary": overview,
                "release_year": release_year,
            }
        except Exception:
            return None


def enrich_movie_event(
    event: Event,
    provider: MovieMetadataProvider,
    logger: StructuredLogger,
) -> Event:
    """Enrich a movie event with metadata from the provider.

    Guards:
        - Only acts on events with source_type in {cinema_veezi, amc}.
        - Skips events with no title.
        - On provider returning None: logs a warning, leaves metadata unchanged.
        - On provider exception: logs an error, leaves metadata unchanged.

    Returns:
        The same Event object (mutated in place if enriched).
    """
    if event.source_type not in _MOVIE_SOURCE_TYPES:
        return event
    if not event.title:
        return event

    try:
        metadata = provider.fetch(event.title, year=None)
        if metadata is None:
            logger.warning(
                f"Movie metadata not found for '{event.title}'",
                component="movies",
            )
            return event
        event.metadata.update(metadata)
    except Exception as exc:
        logger.error(
            f"Movie metadata fetch failed for '{event.title}': {exc}",
            component="movies",
        )

    return event
