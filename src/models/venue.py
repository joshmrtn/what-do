"""Venue data model."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Venue:
    """A physical venue discovered during ingestion or seeding."""

    name: str
    address: str | None = None
    latitude: float | None = None
    longitude: float | None = None
    category: str | None = None
    social_handles: list[str] = field(default_factory=list)
    blocklist_flag: bool = False
    discovery_source: str = ""
