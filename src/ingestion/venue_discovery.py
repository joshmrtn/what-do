"""Venue discovery service — orchestrates providers, dedup, blocklist, and persistence."""

from __future__ import annotations

import json
import math
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rapidfuzz import fuzz

from src.config import AppConfig
from src.ingestion.geocoder import GeocoderProvider
from src.ingestion.seeds import Seeds, load_seeds
from src.ingestion.venue_source import VenueSource
from src.models.venue import Venue


def _haversine_miles(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """Return great-circle distance in miles between two lat/lng points."""
    r = 3958.8
    dlat = math.radians(lat2 - lat1)
    dlng = math.radians(lng2 - lng1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(dlng / 2) ** 2
    )
    return r * 2 * math.asin(math.sqrt(a))


def _fuzzy_match(a: str, b: str) -> float:
    """Return rapidfuzz ratio score (0–100) between two strings, case-insensitive."""
    return fuzz.ratio(a.lower(), b.lower())


class VenueDiscoveryService:
    """Discovers, deduplicates, and persists venues from providers and seeds."""

    def __init__(
        self,
        config: AppConfig,
        db_path: Path,
        seeds_path: Path,
        blocklist_path: Path,
        sources: list[VenueSource],
        geocoder: GeocoderProvider,
        logger: Any,
    ) -> None:
        self._config = config
        self._db_path = db_path
        self._seeds_path = seeds_path
        self._blocklist_path = blocklist_path
        self._sources = sources
        self._geocoder = geocoder
        self._logger = logger

    def run(self) -> None:
        """Execute a full venue discovery pass."""
        blocklist = self._load_blocklist()
        seeds = load_seeds(self._seeds_path)

        conn = sqlite3.connect(self._db_path)
        try:
            self._persist_seed_handles(conn, seeds)
            seed_venues = self._resolve_seed_venues(seeds)
            for venue in seed_venues:
                if self._is_blocked(venue, blocklist):
                    self._logger.info(
                        f"Skipping blocked seed venue: {venue.name}",
                        component="venue_discovery",
                        duration_ms=0,
                    )
                    continue
                self._upsert_venue(conn, venue)

            for source in self._sources:
                try:
                    fetched = source.fetch_venues(
                        lat=self._config.location.latitude,
                        lng=self._config.location.longitude,
                        radius_miles=self._config.location.search_radius_miles,
                        categories=self._config.venue_discovery.categories,
                    )
                except Exception as exc:
                    self._logger.error(
                        f"Venue source {source.__class__.__name__} failed: {exc}",
                        component="venue_discovery",
                        duration_ms=0,
                    )
                    continue

                for venue in fetched:
                    if not self._within_radius(venue):
                        continue
                    if self._is_blocked(venue, blocklist):
                        self._logger.info(
                            f"Skipping blocked venue: {venue.name}",
                            component="venue_discovery",
                            duration_ms=0,
                        )
                        continue
                    self._upsert_venue(conn, venue)

            conn.commit()
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Blocklist
    # ------------------------------------------------------------------

    def _load_blocklist(self) -> list[str]:
        with open(self._blocklist_path) as f:
            return json.load(f)  # type: ignore[no-any-return]

    def _is_blocked(self, venue: Venue, blocklist: list[str]) -> bool:
        threshold = self._config.venue_discovery.blocklist_name_match_threshold * 100
        for entry in blocklist:
            if entry.startswith("@"):
                if entry in venue.social_handles:
                    return True
            else:
                if _fuzzy_match(venue.name, entry) >= threshold:
                    return True
        return False

    # ------------------------------------------------------------------
    # Radius
    # ------------------------------------------------------------------

    def _within_radius(self, venue: Venue) -> bool:
        """Defense-in-depth radius check; venues without coordinates pass through."""
        if venue.latitude is None or venue.longitude is None:
            return True
        dist = _haversine_miles(
            self._config.location.latitude,
            self._config.location.longitude,
            venue.latitude,
            venue.longitude,
        )
        return dist <= self._config.location.search_radius_miles

    # ------------------------------------------------------------------
    # Seeds
    # ------------------------------------------------------------------

    def _resolve_seed_venues(self, seeds: Seeds) -> list[Venue]:
        """Geocode seed venue entries and return them as Venue objects."""
        resolved: list[Venue] = []
        for sv in seeds.venues:
            lat: float | None = None
            lng: float | None = None
            try:
                result = self._geocoder.geocode(sv.address)
                if result is None:
                    self._logger.warning(
                        f"Geocoding returned no result for seed venue address: {sv.address}",
                        component="venue_discovery",
                        duration_ms=0,
                    )
                else:
                    lat, lng = result
            except Exception as exc:
                self._logger.warning(
                    f"Geocoding failed for seed venue '{sv.name}': {exc}",
                    component="venue_discovery",
                    duration_ms=0,
                )
            resolved.append(
                Venue(
                    name=sv.name,
                    address=sv.address,
                    latitude=lat,
                    longitude=lng,
                    discovery_source="seed",
                )
            )
        return resolved

    def _persist_seed_handles(self, conn: sqlite3.Connection, seeds: Seeds) -> None:
        now = datetime.now(timezone.utc).isoformat()
        for handle in seeds.handles:
            existing = conn.execute(
                "SELECT id FROM candidate_entities WHERE handle = ?", (handle,)
            ).fetchone()
            if existing:
                conn.execute(
                    "UPDATE candidate_entities SET state = 'active', depth = 0, updated_at = ? WHERE handle = ?",
                    (now, handle),
                )
                continue
            conn.execute(
                """INSERT INTO candidate_entities
                   (id, handle, state, depth, mention_count, mention_sources, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (str(uuid.uuid4()), handle, "active", 0, 0, "[]", now, now),
            )

    # ------------------------------------------------------------------
    # Persistence / dedup
    # ------------------------------------------------------------------

    def _upsert_venue(self, conn: sqlite3.Connection, venue: Venue) -> None:
        """Insert venue if no near-duplicate exists; skip if duplicate detected."""
        name_threshold = self._config.venue_discovery.name_match_threshold * 100
        addr_threshold = self._config.venue_discovery.address_match_threshold * 100

        existing = conn.execute(
            "SELECT id, name, address FROM venues"
        ).fetchall()

        for eid, ename, eaddress in existing:
            name_score = _fuzzy_match(venue.name, ename)
            if name_score < name_threshold:
                continue
            if venue.address and eaddress:
                addr_score = _fuzzy_match(venue.address, eaddress)
                if addr_score < addr_threshold:
                    continue
            # Duplicate found — skip
            return

        now = datetime.now(timezone.utc).isoformat()
        conn.execute(
            """INSERT INTO venues
               (id, name, address, latitude, longitude, category,
                social_handles, blocklisted, discovery_source, discovered_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                str(uuid.uuid4()),
                venue.name,
                venue.address,
                venue.latitude,
                venue.longitude,
                venue.category,
                json.dumps(venue.social_handles),
                1 if venue.blocklist_flag else 0,
                venue.discovery_source,
                now,
            ),
        )
