"""Movie metadata provider ABC, TMDb implementation, and event enrichment helper."""

from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Any, Callable

import requests

from src.config import ConfigError
from src.models.event import Event
from src.models.movie_lookup import MovieLookup
from src.network.http import requests_transient_check
from src.network.policy import RequestPolicy
from src.storage.protocols import MovieCache
from src.utils.logging import StructuredLogger
from src.utils.secret import Secret

#: The host TMDb's API is reached at, named so a caller can look up its
#: politeness policy without owning a second copy of the address.
TMDB_HOST = "api.themoviedb.org"


def title_key(title: str) -> str:
    """The canonical form two spellings of one film share.

    Compare on a canonical key; store what the source wrote. Casefolded because
    one cinema publishes `BACK TO THE FUTURE` and another `Back to the Future`,
    and whitespace collapsed because listings pad and wrap titles. It is
    deliberately *not* what gets stored or sent to TMDb — the query keeps the
    title as the listing wrote it.

    Nothing is stripped beyond case and spacing: `The Thing` and `The Thing II`
    are different films, and an article rule that folded them would be worse
    than no rule at all.
    """
    return " ".join(title.split()).casefold()

_MOVIE_SOURCE_TYPES = {"cinema_veezi", "amc"}


class MovieMetadataProvider(ABC):
    """Abstract base for movie metadata providers."""

    @abstractmethod
    def fetch(self, title: str, year: int | None) -> dict[str, Any] | None:
        """Fetch metadata for a movie title.

        Returns:
            Dict with keys genres, runtime_minutes, summary, release_year,
            or None if the movie was not found.
        """


class TMDbProvider(MovieMetadataProvider):
    """Movie metadata provider backed by The Movie Database (TMDb) API.

    **Authenticates with a v4 read access token in an `Authorization` header.**
    TMDb's older v3 key can only travel as `?api_key=`, and support for it is
    removed rather than kept as a fallback: keeping it would mean keeping a
    credential in a URL and relying on log scrubbing to hide it, which is a
    backstop standing in for a fix. A value that never enters the URL cannot
    leak through a surface nobody predicted.
    """

    _BASE_URL = f"https://{TMDB_HOST}/3"

    def __init__(
        self,
        read_access_token: Secret,
        *,
        session: requests.Session,
        policy: RequestPolicy,
        movie_cache: MovieCache,
        cache_ttl: timedelta | None,
        get_now: Callable[[], datetime],
    ) -> None:
        """
        Args:
            read_access_token: TMDb credential. Named for the label TMDb itself
                puts on it, so the parameter matches the page the value is
                copied from — and because a parameter called `api_key` is what
                made the query-string form look natural in the first place.
            session: Injected HTTP session, so tests never reach the network.
            policy: Throttle, retry and timeout for this host.
            movie_cache: Answers keyed on canonical title and year, **misses
                included**.
            cache_ttl: Lifetime from the `tmdb` policy. `None` is a declared
                `never`.
            get_now: Injected clock.
        """
        self._token = read_access_token
        self._session = session
        self._policy = policy
        self._cache = movie_cache
        self._cache_ttl = cache_ttl
        self._get_now = get_now
        self._is_transient = requests_transient_check(get_now=get_now)

    def fetch(self, title: str, year: int | None) -> dict[str, Any] | None:
        """Search TMDb for a movie and return structured metadata.

        Returns:
            Dict with genres, runtime_minutes, summary, release_year, or None
            when TMDb was asked and had nothing.

        A not-found is an **answer**, and it is cached like any other: a cinema
        listing is full of titles TMDb does not recognise, and a miss that is
        not stored is a request that repeats on every run for ever. A *failure*
        raises instead, so the policy can judge whether to try again.

        The catch is narrow on purpose. `except Exception` here would report a
        bug in this file as a film nobody has heard of.
        """
        try:
            lookup = self._policy.call(
                host=TMDB_HOST,
                perform=lambda timeout: self._request(title, year, timeout),
                is_transient=self._is_transient,
                cache=MovieTitleCache(
                    self._cache,
                    title_key=title_key(title),
                    year=year,
                    ttl=self._cache_ttl,
                    get_now=self._get_now,
                ),
                label="tmdb",
            )
        except ConfigError:
            # `ConfigError` is a `ValueError`, so the narrow catch below would
            # swallow an unassigned host as an absence — and an absence here is
            # indistinguishable from a provider that had nothing to say. That is
            # the swallow this catch was narrowed to prevent, arriving by
            # inheritance instead of by breadth.
            raise
        except (requests.RequestException, ValueError, KeyError):
            return None
        return lookup.metadata

    def _request(self, title: str, year: int | None, timeout: float) -> MovieLookup:
        """One attempt: search, then detail. Raises so the policy can retry.

        Two requests, one lookup. They are a single question — *what is this
        film* — and the throttle spaces both because it counts per host.
        """
        params: dict[str, str | int] = {"query": title}
        if year is not None:
            params["year"] = year
        headers = {"Authorization": f"Bearer {self._token.expose_secret()}"}

        search_resp = self._session.get(
            f"{self._BASE_URL}/search/movie",
            params=params,
            headers=headers,
            timeout=timeout,
        )
        search_resp.raise_for_status()
        results = search_resp.json().get("results", [])
        if not results:
            return MovieLookup(metadata=None)

        movie_id = results[0]["id"]
        release_date: str = results[0].get("release_date", "")

        detail_resp = self._session.get(
            f"{self._BASE_URL}/movie/{movie_id}",
            headers=headers,
            timeout=timeout,
        )
        detail_resp.raise_for_status()
        detail = detail_resp.json()

        genres = [g["name"] for g in detail.get("genres", [])]
        runtime = detail.get("runtime")
        overview = detail.get("overview") or None
        release_year_str = (detail.get("release_date") or release_date)[:4]
        release_year = int(release_year_str) if release_year_str.isdigit() else None

        return MovieLookup(
            metadata={
                "genres": genres,
                "runtime_minutes": int(runtime) if runtime else None,
                "summary": overview,
                "release_year": release_year,
            }
        )


class MovieTitleCache:
    """A `CacheStrategy[MovieLookup]` bound to one title and year."""

    def __init__(
        self,
        cache: MovieCache,
        *,
        title_key: str,
        year: int | None,
        ttl: timedelta | None,
        get_now: Callable[[], datetime],
    ) -> None:
        self._cache = cache
        self._title_key = title_key
        self._year = year
        self._ttl = ttl
        self._get_now = get_now

    def get(self) -> MovieLookup | None:
        """The stored answer while it is still fresh, otherwise None.

        None here means *nothing stored*, never *nothing found* — a stored miss
        comes back as a `MovieLookup` with no metadata, which is what stops it
        being asked again.
        """
        if self._ttl is None:
            return None
        return self._cache.get(
            title_key=self._title_key,
            year=self._year,
            fresh_since=self._get_now() - self._ttl,
        )

    def put(self, value: MovieLookup) -> None:
        """Store the answer, misses included, stamped from the injected clock."""
        if self._ttl is None:
            return
        self._cache.put(
            title_key=self._title_key,
            year=self._year,
            lookup=value,
            now=self._get_now(),
        )


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
