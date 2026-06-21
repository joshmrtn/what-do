"""Normalization engine — converts EventCandidates into canonical Events."""

from __future__ import annotations

import re
import unicodedata
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Callable
from zoneinfo import ZoneInfo

from src.models.event import Event
from src.models.event_candidate import EventCandidate

_ARTICLE_SUFFIX_RE = re.compile(r"^(.*),\s*(the|a|an)$", re.IGNORECASE)


def _normalize_text(text: str | None) -> str | None:
    """NFC-normalize, replace non-breaking spaces, collapse whitespace."""
    if text is None:
        return None
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\xa0", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _normalize_venue(name: str | None) -> str | None:
    """Canonicalize venue name: move trailing article to front, title-case."""
    if name is None:
        return None
    name = re.sub(r"\s+", " ", name.replace("\xa0", " ")).strip()
    match = _ARTICLE_SUFFIX_RE.match(name)
    if match:
        body, article = match.group(1).strip(), match.group(2)
        name = f"{article.capitalize()} {body}"
    return name.title()


def _normalize_timestamp(dt: datetime | None, tz: ZoneInfo) -> datetime | None:
    """Convert timezone-aware dt to config tz; attach tz to naive dt."""
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=tz)
    return dt.astimezone(tz)


@dataclass
class DiscardedCandidate:
    """A candidate that was discarded during normalization, with its reason."""

    candidate: EventCandidate
    reason: str


@dataclass
class _NormalizerOutput:
    """Internal result from NormalizationEngine.normalize()."""

    events: list[Event]
    discards: list[DiscardedCandidate]


class NormalizationEngine:
    """Convert a list of EventCandidates into normalized Events.

    Pure — no I/O, no DB access. Discarded candidates are returned with
    their reason; the caller is responsible for logging them.
    """

    def __init__(self, timezone_name: str) -> None:
        """
        Args:
            timezone_name: IANA timezone string derived from config coordinates.
        """
        self._tz = ZoneInfo(timezone_name)

    def normalize(
        self,
        candidates: list[EventCandidate],
        get_now: Callable[[], datetime] = datetime.now,
    ) -> _NormalizerOutput:
        """Normalize a list of EventCandidates into Events.

        Args:
            candidates: Raw candidates from the ingestion layer.
            get_now: Injectable clock for created_at / updated_at timestamps.

        Returns:
            _NormalizerOutput with valid events and a list of discarded candidates.
        """
        events: list[Event] = []
        discards: list[DiscardedCandidate] = []
        now = get_now()

        for candidate in candidates:
            event = self._normalize_one(candidate, now)
            if event is None:
                discards.append(DiscardedCandidate(
                    candidate=candidate,
                    reason="missing both title and start_time",
                ))
            else:
                events.append(event)

        return _NormalizerOutput(events=events, discards=discards)

    def _normalize_one(self, candidate: EventCandidate, now: datetime) -> Event | None:
        title = _normalize_text(candidate.title)
        start_time = _normalize_timestamp(candidate.start_time, self._tz)

        if title is None and start_time is None:
            return None

        metadata: dict = {}
        if title is None:
            metadata["missing_title"] = True
        if start_time is None:
            metadata["missing_start_time"] = True

        return Event(
            event_id=str(uuid.uuid4()),
            source_event_candidates=[candidate.id],
            source_type=candidate.source_type,
            url=candidate.url,
            image_url=candidate.image_url,
            title=title,
            venue=_normalize_venue(candidate.venue),
            description=_normalize_text(candidate.description),
            location=_normalize_text(candidate.location),
            start_time=start_time,
            end_time=_normalize_timestamp(candidate.end_time, self._tz),
            metadata=metadata,
            created_at=now,
            updated_at=now,
        )
