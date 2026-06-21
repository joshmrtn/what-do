"""Loader for data/seeds.yaml — the bootstrap source list."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class SeedVenueEntry:
    """A manually known venue from seeds.yaml."""

    name: str
    address: str


@dataclass
class Seeds:
    """Parsed contents of seeds.yaml."""

    handles: list[str] = field(default_factory=list)
    venues: list[SeedVenueEntry] = field(default_factory=list)


def load_seeds(path: Path) -> Seeds:
    """Parse seeds.yaml into a Seeds object.

    Args:
        path: Path to seeds.yaml.

    Returns:
        Seeds with handles and venue entries.
    """
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    handles = [str(h) for h in data.get("handles", [])]
    venues = [
        SeedVenueEntry(name=v["name"], address=v["address"])
        for v in data.get("venues", [])
    ]
    return Seeds(handles=handles, venues=venues)
