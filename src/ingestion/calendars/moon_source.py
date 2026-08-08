"""MOON's show feed, read through the RSS source.

MOON is a promoter rather than a venue: its shows run at Felt Fanatic, Bit Bar,
Faces and elsewhere, so the venue belongs to the item and not to the feed.

Its convention, measured across the feed's twenty items, is that the event date
lives in the title as `M/D/YY` or `M/D/YYYY` — not always at the front, since
titles carry markers like `*** Moved to Faces Brewery ***` ahead of it. Seventeen
of twenty items carry one; the three that do not carry no date anywhere and are
refused rather than guessed at.

`pubDate` is never the event date. Every item in the feed was posted in May for a
June show.
"""

from __future__ import annotations

import re
from datetime import datetime

from src.ingestion.calendars.rss_source import RssEvent, RssFeedSource, looks_cancelled
from src.ingestion.rss import RssItem
from src.models.timing import EXACT, UNKNOWN
from src.utils.html import html_to_text

#: `6/27/26` or `6/5/2026`, anywhere in the title.
_DATE_RE = re.compile(r"\b(\d{1,2})/(\d{1,2})/(\d{4}|\d{2})\b")

#: The same, but only where it opens the title, with its separator.
_LEADING_DATE_RE = re.compile(r"^\s*\d{1,2}/\d{1,2}/(?:\d{4}|\d{2})\s*[:\-–]?\s*")

#: `at 6pm`, `at 7:30 p.m.` — the description's own phrasing for the hour.
_TIME_RE = re.compile(r"\bat\s+(\d{1,2})(?::(\d{2}))?\s*([ap])\.?m\.?", re.IGNORECASE)

#: What a venue name may look like. The text after `at <time> at ` is a venue on
#: a templated item and a run of prose on a hand-written one — `Faces Malden -
#: ALL AGES | $15 Day of Show` is not a place, and no venue beats a wrong one.
_VENUE_RE = re.compile(r"^[A-Za-z0-9'&.\- ]{2,40}$")


class MoonRssSource(RssFeedSource):
    """Reads MOON's `/shows?format=rss` feed."""

    def interpret(self, item: RssItem) -> RssEvent | None:
        """Read one show out of a feed item.

        Args:
            item: The parsed `<item>`.

        Returns:
            The show, or None if it carries no date or has been cancelled.
        """
        if looks_cancelled(item.title):
            return None

        match = _DATE_RE.search(item.title)
        if match is None:
            return None

        body = " ".join(html_to_text(item.description or "").split())
        start = self._place(match, body)
        if start is None:
            return None

        return RssEvent(
            title=_LEADING_DATE_RE.sub("", item.title).strip() or item.title,
            start=start,
            timing=EXACT if _TIME_RE.search(body) else UNKNOWN,
            venue=_venue_in(body),
            description=body or None,
        )

    def _place(self, date_match: re.Match[str], body: str) -> datetime | None:
        """Build the start from the title's date and the description's hour."""
        month, day, year = (int(part) for part in date_match.groups())
        if year < 100:
            year += 2000

        hour, minute = self._hour_in(body)
        try:
            return datetime(year, month, day, hour, minute, tzinfo=self.zone)
        except ValueError:
            # `13/40/26` is not a date. Nothing here can repair one.
            return None

    def _hour_in(self, body: str) -> tuple[int, int]:
        """The stated hour, or the day's start when nobody published one."""
        match = _TIME_RE.search(body)
        if match is None:
            return self.day_starts_at.hour, self.day_starts_at.minute

        hour = int(match.group(1)) % 12
        if match.group(3).lower() == "p":
            hour += 12

        return hour, int(match.group(2) or 0)


def _venue_in(body: str) -> str | None:
    """The venue the description's template names, if it names one cleanly."""
    match = _TIME_RE.search(body)
    if match is None:
        return None

    tail = re.sub(r"^at\s+", "", body[match.end() :].strip(), flags=re.IGNORECASE).strip()

    return tail if tail and _VENUE_RE.match(tail) else None
