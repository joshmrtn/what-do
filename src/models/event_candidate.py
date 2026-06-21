"""EventCandidate data model — raw discovered event information."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass
class EventCandidate:
    """Raw event information as returned by an ingestion source adapter."""

    id: str
    source: str
    source_type: str
    discovered_at: datetime
    url: str | None = None
    image_url: str | None = None
    raw_published_at: datetime | None = None
    title: str | None = None
    description: str | None = None
    venue: str | None = None
    location: str | None = None
    start_time: datetime | None = None
    end_time: datetime | None = None
