"""Veezi public ticketing source adapter.

Reads a cinema's public `sessions` page and maps each showing to an
EventCandidate. Like the calendar sources this is forward-looking and already
structured, so title, time and booking link come out without an LLM.

No credentials. The page is keyed by a `siteToken` that appears in the cinema's
own booking links, and one adapter therefore covers every Veezi cinema — the
token is part of the configured URL, not code.

Politeness matches the calendar adapters: one conditional request per
`min_fetch_interval_hours`, cached across process restarts, and no retry loop.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Callable
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from src.config import FeedConfig
from src.network.http import HttpFetcher
from src.ingestion.candidate_id import derive_content_id
from src.ingestion.cinemas.veezi_listing import VeeziSession, parse_sessions
from src.ingestion.identity import ContentIdRule
from src.ingestion.source import IngestionSource
from src.models.event_candidate import EventCandidate


class VeeziSessionsSource(IngestionSource):
    """Fetches event candidates from a Veezi public sessions page."""

    def __init__(
        self,
        config: FeedConfig,
        fetcher: HttpFetcher,
        get_now: Callable[[], datetime] = datetime.now,
        logger: Any = None,
        timezone_name: str = "UTC",
        *,
        uses_content_id: ContentIdRule,
    ) -> None:
        self._config = config
        self._fetcher = fetcher
        self._get_now = get_now
        self._logger = logger
        self._zone = _zone_of(timezone_name)
        # Required and keyword-only, so the composition root cannot forget it.
        self._uses_content_id = uses_content_id

    @property
    def source_name(self) -> str:
        """The feed's configured name, so a report can name the feed not the class."""
        return self._config.name

    def fetch(self) -> list[EventCandidate]:
        """Fetch and parse the sessions page, skipping the network when polite to.

        Returns:
            One EventCandidate per showing, in page order. Bounding by event time
            is left to ingestion, which applies the same window to every source.
        """
        body = self._fetcher.get(
            self._config.url,
            label=self._config.name,
            max_age=timedelta(hours=self._config.min_fetch_interval_hours),
        )

        today = self._get_now().astimezone(self._zone).date()
        sessions = parse_sessions(body, today, logger=self._logger)

        return [self._to_candidate(showing) for showing in sessions]

    def _to_candidate(self, showing: VeeziSession) -> EventCandidate:
        """Map one showing, localising its wall clock to the cinema's zone."""
        start = showing.start.replace(tzinfo=self._zone)
        return EventCandidate(
            id=self._candidate_id(showing, start),
            source=self._config.name,
            source_type=self._config.source_type,
            title=showing.title,
            description=None,
            # The page names the film and nothing else; only config knows which
            # cinema this is.
            venue=self._config.venue,
            location=self._config.city,
            url=showing.url,
            start_time=start,
            # Runtime is not published here. TMDb enrichment can supply it later
            # rather than the adapter inventing a duration.
            end_time=None,
            # Deliberately unset, as for the calendar sources: a showtime
            # listing carries no announcement date, and this field is what the
            # ingestion lookback discards on.
            raw_published_at=None,
            discovered_at=self._get_now(),
        )

    def _candidate_id(self, showing: VeeziSession, start: datetime) -> str:
        """The session id, unless this cinema's session ids identify nothing.

        Veezi publishes a `session_id` and it is the better key while it holds —
        it survives a retitling, and it tells one screen from another where a
        content key cannot. The content path exists for when it stops holding.
        """
        if self._uses_content_id(self._config.name):
            return derive_content_id(
                source=self._config.name,
                title=showing.title,
                venue=self._config.venue,
                start=start,
            )

        return f"{self._config.name}:{showing.session_id}"


def _zone_of(name: str) -> ZoneInfo:
    """Resolve the cinema's zone, falling back to UTC rather than failing a run."""
    try:
        return ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError):
        return ZoneInfo("UTC")
