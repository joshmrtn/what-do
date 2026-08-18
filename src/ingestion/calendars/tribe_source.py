"""The Events Calendar (WordPress) iCal source.

The plugin's iCal export is capped at **30 events**, whatever the calendar
holds. `paged` and `tribe_paged` are ignored; `tribe-bar-date=YYYY-MM-DD` moves
the window, so the export is walked by date rather than by page.

Without the walk a busy calendar silently contributes its first 30 events and
looks complete. Measured: one aggregator's export covered barely two days.

Everything else — parsing, recurrence expansion, the event-time window, venue
defaults — is `IcsCalendarSource`'s. This overrides only how many documents make
up the calendar.
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta
from typing import Any, Callable

from src.config import DEFAULT_DAY_STARTS_AT, DEFAULT_HORIZON_DAYS, FeedConfig
from src.network.http import HttpFetcher
from src.ingestion.calendars.ics_source import IcsCalendarSource
from src.ingestion.ics import parse_ics
from src.utils.nights import night_start

#: Never make more than this many requests for one calendar, whatever it
#: reports. A calendar that always names a later date must not become a crawl.
_DEFAULT_MAX_REQUESTS = 8

#: How many events the plugin will export at once. A window returning fewer
#: than this is the last one, which saves a request on every short calendar.
#: Measured at 30 across four independent sites; injectable because it is
#: their number, not ours.
_DEFAULT_EXPORT_CAP = 30


class TribeCalendarSource(IcsCalendarSource):
    """Fetches a The Events Calendar iCal export, walking past its 30-event cap."""

    def __init__(
        self,
        config: FeedConfig,
        fetcher: HttpFetcher,
        get_now: Callable[[], datetime] = datetime.now,
        logger: Any = None,
        timezone_name: str = "UTC",
        horizon_days: int = DEFAULT_HORIZON_DAYS,
        day_starts_at: time = DEFAULT_DAY_STARTS_AT,
        max_requests: int = _DEFAULT_MAX_REQUESTS,
        export_cap: int = _DEFAULT_EXPORT_CAP,
    ) -> None:
        super().__init__(
            config,
            fetcher,
            get_now=get_now,
            logger=logger,
            timezone_name=timezone_name,
            horizon_days=horizon_days,
            day_starts_at=day_starts_at,
        )
        self._max_requests = max_requests
        self._export_cap = export_cap

    def _read_documents(self) -> list[str]:
        """Walk `tribe-bar-date` until the calendar is exhausted or out of range."""
        horizon = (
            night_start(self._get_now(), self._day_starts_at, self._zone)
            + timedelta(days=self._horizon_days)
        ).date()

        documents: list[str] = []
        seen: set[str | None] = set()
        bar_date: date | None = None

        for _ in range(self._max_requests):
            body = self._read_feed(self._url_for(bar_date))
            documents.append(body)

            events = parse_ics(body)
            uids = {event.uid for event in events}
            if not uids - seen:
                # Either the window is empty or it repeated what we already
                # hold, both of which mean there is nothing further to ask for.
                break
            seen |= uids

            if len(events) < self._export_cap:
                # Short of the cap, so the export had nothing more to give.
                break

            latest = max(
                (event.dtstart.date() for event in events if event.dtstart), default=None
            )
            if latest is None or latest > horizon:
                # The window already reaches past everything this run will rank.
                break
            if bar_date is not None and latest <= bar_date:
                # The walk stopped advancing; asking again returns the same window.
                break
            bar_date = latest

        return documents

    def _url_for(self, bar_date: date | None) -> str:
        if bar_date is None:
            return self._config.url

        separator = "&" if "?" in self._config.url else "?"

        return f"{self._config.url}{separator}tribe-bar-date={bar_date.isoformat()}"
