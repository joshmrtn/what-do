"""Unit tests for venue discovery service."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import yaml

from src.config import AppConfig, LocationConfig, ScrapingConfig, VenueDiscoveryConfig
from src.ingestion.geocoder import GeocoderProvider
from src.ingestion.venue_discovery import VenueDiscoveryService
from src.ingestion.venue_source import VenueSource
from src.models.venue import Venue
from src.storage.db import init_db
from src.utils.logging import get_logger

# ~1 mile north of Salem city center — well within any reasonable radius
SALEM_LAT = 42.5195
SALEM_LNG = -70.8967


# ---- Helpers -----------------------------------------------------------------


def _make_config(
    lat: float = SALEM_LAT,
    lng: float = SALEM_LNG,
    radius: float = 10.0,
    categories: list[str] | None = None,
    name_threshold: float = 0.92,
    address_threshold: float = 0.85,
    blocklist_threshold: float = 0.80,
) -> AppConfig:
    return AppConfig(
        location=LocationConfig(
            latitude=lat,
            longitude=lng,
            postal_code="01970",
            search_radius_miles=radius,
            timezone="America/New_York",
        ),
        scraping=ScrapingConfig(),
        venue_discovery=VenueDiscoveryConfig(
            categories=categories or ["cafe", "bar", "music_venue"],
            name_match_threshold=name_threshold,
            address_match_threshold=address_threshold,
            blocklist_name_match_threshold=blocklist_threshold,
        ),
        ollama_host="http://localhost:11434",
    )


def _make_service(
    db_path: Path,
    seeds_path: Path,
    blocklist_path: Path,
    sources: list[VenueSource] | None = None,
    geocoder: GeocoderProvider | None = None,
    config: AppConfig | None = None,
    logger: object | None = None,
) -> VenueDiscoveryService:
    if geocoder is None:
        geocoder = MagicMock(spec=GeocoderProvider)
        geocoder.geocode.return_value = None
    return VenueDiscoveryService(
        config=config or _make_config(),
        db_path=db_path,
        seeds_path=seeds_path,
        blocklist_path=blocklist_path,
        sources=sources or [],
        geocoder=geocoder,
        logger=logger or get_logger("test"),
    )


def _venue_rows(db_path: Path) -> list[tuple]:
    conn = sqlite3.connect(db_path)
    rows = conn.execute("SELECT * FROM venues").fetchall()
    conn.close()
    return rows


def _venue_count(db_path: Path) -> int:
    return len(_venue_rows(db_path))


def _candidate_handles(db_path: Path) -> list[str]:
    conn = sqlite3.connect(db_path)
    rows = conn.execute("SELECT handle FROM candidate_entities").fetchall()
    conn.close()
    return [r[0] for r in rows]


def _near_venue(name: str = "Near Venue", address: str = "1 Derby St") -> Venue:
    return Venue(
        name=name,
        address=address,
        latitude=SALEM_LAT + 0.005,
        longitude=SALEM_LNG,
        category="cafe",
        discovery_source="test",
    )


# ---- Fixtures ----------------------------------------------------------------


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    path = tmp_path / "test.db"
    init_db(path)
    return path


@pytest.fixture
def seeds_path(tmp_path: Path) -> Path:
    path = tmp_path / "seeds.yaml"
    path.write_text(yaml.dump({"handles": [], "venues": []}))
    return path


@pytest.fixture
def empty_blocklist(tmp_path: Path) -> Path:
    path = tmp_path / "blocklist.json"
    path.write_text("[]")
    return path


# ---- Provider abstraction ----------------------------------------------------


class TestProviderAbstraction:
    def test_any_venue_source_implementation_works_without_service_changes(
        self, db_path: Path, seeds_path: Path, empty_blocklist: Path
    ) -> None:
        """Swapping VenueSource implementations requires no changes to the service."""

        class CustomSource(VenueSource):
            def fetch_venues(
                self, lat: float, lng: float, radius_miles: float, categories: list[str]
            ) -> list[Venue]:
                return [_near_venue("Custom Venue")]

        svc = _make_service(db_path, seeds_path, empty_blocklist, sources=[CustomSource()])
        svc.run()
        assert _venue_count(db_path) == 1

    def test_failing_source_is_skipped_and_remaining_sources_run(
        self, db_path: Path, seeds_path: Path, empty_blocklist: Path
    ) -> None:
        """A provider that raises is logged and skipped; other providers still run."""
        failing = MagicMock(spec=VenueSource)
        failing.fetch_venues.side_effect = RuntimeError("connection refused")

        good = MagicMock(spec=VenueSource)
        good.fetch_venues.return_value = [_near_venue("Good Venue")]

        errors: list[str] = []
        logger = MagicMock()
        logger.error = lambda msg, **kw: errors.append(msg)

        svc = _make_service(
            db_path, seeds_path, empty_blocklist,
            sources=[failing, good],
            logger=logger,
        )
        svc.run()

        assert _venue_count(db_path) == 1
        assert errors, "expected error to be logged for the failing provider"


# ---- Radius filtering --------------------------------------------------------


class TestRadiusFiltering:
    def test_venue_within_radius_is_persisted(
        self, db_path: Path, seeds_path: Path, empty_blocklist: Path
    ) -> None:
        source = MagicMock(spec=VenueSource)
        source.fetch_venues.return_value = [_near_venue()]

        _make_service(db_path, seeds_path, empty_blocklist, sources=[source]).run()
        assert _venue_count(db_path) == 1

    def test_venue_outside_radius_is_excluded(
        self, db_path: Path, seeds_path: Path, empty_blocklist: Path
    ) -> None:
        """Provider result outside the configured radius is not persisted."""
        source = MagicMock(spec=VenueSource)
        # Boston is ~15 miles from Salem — outside a 10-mile radius
        source.fetch_venues.return_value = [
            Venue(
                name="Boston Venue",
                address="1 Boylston St",
                latitude=42.3601,
                longitude=-71.0589,
                category="cafe",
                discovery_source="test",
            )
        ]

        _make_service(db_path, seeds_path, empty_blocklist, sources=[source]).run()
        assert _venue_count(db_path) == 0

    def test_provider_receives_configured_radius(
        self, db_path: Path, seeds_path: Path, empty_blocklist: Path
    ) -> None:
        source = MagicMock(spec=VenueSource)
        source.fetch_venues.return_value = []
        cfg = _make_config(radius=7.5)

        _make_service(db_path, seeds_path, empty_blocklist, sources=[source], config=cfg).run()

        args, kwargs = source.fetch_venues.call_args
        radius = kwargs.get("radius_miles") if kwargs else args[2]
        assert radius == 7.5


# ---- Schema completeness -----------------------------------------------------


class TestVenueSchema:
    def test_discovered_venue_stored_with_all_schema_fields(
        self, db_path: Path, seeds_path: Path, empty_blocklist: Path
    ) -> None:
        source = MagicMock(spec=VenueSource)
        source.fetch_venues.return_value = [
            Venue(
                name="The Vault",
                address="1 Pickering Wharf",
                latitude=SALEM_LAT,
                longitude=SALEM_LNG,
                category="music_venue",
                social_handles=["@thevaultlounge"],
                discovery_source="test_provider",
            )
        ]

        _make_service(db_path, seeds_path, empty_blocklist, sources=[source]).run()

        conn = sqlite3.connect(db_path)
        row = conn.execute(
            "SELECT name, address, latitude, longitude, category, social_handles, discovery_source FROM venues"
        ).fetchone()
        conn.close()

        assert row[0] == "The Vault"
        assert row[1] == "1 Pickering Wharf"
        assert row[2] == pytest.approx(SALEM_LAT)
        assert row[3] == pytest.approx(SALEM_LNG)
        assert row[4] == "music_venue"
        assert "@thevaultlounge" in row[5]
        assert row[6] == "test_provider"


# ---- Deduplication -----------------------------------------------------------


class TestDeduplication:
    def test_identical_venue_from_two_sources_produces_one_record(
        self, db_path: Path, seeds_path: Path, empty_blocklist: Path
    ) -> None:
        v = _near_venue("The Tap Room", "12 Derby St")
        source_a = MagicMock(spec=VenueSource)
        source_a.fetch_venues.return_value = [v]
        source_b = MagicMock(spec=VenueSource)
        source_b.fetch_venues.return_value = [_near_venue("The Tap Room", "12 Derby St")]

        _make_service(
            db_path, seeds_path, empty_blocklist, sources=[source_a, source_b]
        ).run()
        assert _venue_count(db_path) == 1

    def test_same_name_different_address_are_distinct(
        self, db_path: Path, seeds_path: Path, empty_blocklist: Path
    ) -> None:
        """Same chain name at different addresses must not be merged."""
        source = MagicMock(spec=VenueSource)
        source.fetch_venues.return_value = [
            Venue(
                name="Holy Cow Ice Cream",
                address="1 Main St, Peabody MA",
                latitude=SALEM_LAT + 0.01,
                longitude=SALEM_LNG,
                category="cafe",
                discovery_source="test",
            ),
            Venue(
                name="Holy Cow Ice Cream",
                address="999 Washington St, Gloucester MA",
                latitude=SALEM_LAT + 0.02,
                longitude=SALEM_LNG,
                category="cafe",
                discovery_source="test",
            ),
        ]

        _make_service(db_path, seeds_path, empty_blocklist, sources=[source]).run()
        assert _venue_count(db_path) == 2

    def test_running_discovery_twice_does_not_duplicate(
        self, db_path: Path, seeds_path: Path, empty_blocklist: Path
    ) -> None:
        source = MagicMock(spec=VenueSource)
        source.fetch_venues.return_value = [_near_venue("The Tap Room", "12 Derby St")]
        svc = _make_service(db_path, seeds_path, empty_blocklist, sources=[source])

        svc.run()
        svc.run()
        assert _venue_count(db_path) == 1


# ---- Blocklist enforcement ---------------------------------------------------


class TestBlocklist:
    def test_venue_with_blocked_handle_is_skipped_and_logged(
        self, db_path: Path, tmp_path: Path, seeds_path: Path
    ) -> None:
        bl = tmp_path / "blocklist.json"
        bl.write_text(json.dumps(["@badplace"]))

        source = MagicMock(spec=VenueSource)
        source.fetch_venues.return_value = [
            Venue(
                name="Bad Place",
                address="1 Hex St",
                latitude=SALEM_LAT,
                longitude=SALEM_LNG,
                category="bar",
                social_handles=["@badplace"],
                discovery_source="test",
            )
        ]

        log_msgs: list[str] = []
        logger = MagicMock()
        logger.info = lambda msg, **kw: log_msgs.append(msg)

        _make_service(db_path, seeds_path, bl, sources=[source], logger=logger).run()

        assert _venue_count(db_path) == 0
        assert any("block" in m.lower() for m in log_msgs)

    def test_venue_with_fuzzy_blocked_name_is_skipped(
        self, db_path: Path, tmp_path: Path, seeds_path: Path
    ) -> None:
        """A venue name that fuzzy-matches a blocklist entry is excluded."""
        bl = tmp_path / "blocklist.json"
        bl.write_text(json.dumps(["O'Neils Bar"]))

        source = MagicMock(spec=VenueSource)
        source.fetch_venues.return_value = [
            Venue(
                name="O'Neils Bar",  # exact match to confirm threshold works
                address="5 Pickering",
                latitude=SALEM_LAT,
                longitude=SALEM_LNG,
                category="bar",
                discovery_source="test",
            )
        ]

        _make_service(db_path, seeds_path, bl, sources=[source]).run()
        assert _venue_count(db_path) == 0

    def test_non_blocked_venue_is_not_excluded(
        self, db_path: Path, tmp_path: Path, seeds_path: Path
    ) -> None:
        bl = tmp_path / "blocklist.json"
        bl.write_text(json.dumps(["@someotherplace"]))

        source = MagicMock(spec=VenueSource)
        source.fetch_venues.return_value = [_near_venue("The Good Spot")]

        _make_service(db_path, seeds_path, bl, sources=[source]).run()
        assert _venue_count(db_path) == 1


# ---- Seed integration --------------------------------------------------------


class TestSeedIntegration:
    def test_seed_venue_written_to_venues_table_with_source_seed(
        self, db_path: Path, tmp_path: Path, empty_blocklist: Path
    ) -> None:
        seeds = tmp_path / "seeds.yaml"
        seeds.write_text(
            yaml.dump(
                {
                    "handles": [],
                    "venues": [{"name": "Cinema Salem", "address": "95 Washington St"}],
                }
            )
        )
        geocoder = MagicMock(spec=GeocoderProvider)
        geocoder.geocode.return_value = (SALEM_LAT, SALEM_LNG)

        _make_service(db_path, seeds, empty_blocklist, geocoder=geocoder).run()

        conn = sqlite3.connect(db_path)
        row = conn.execute(
            "SELECT name, discovery_source FROM venues WHERE name='Cinema Salem'"
        ).fetchone()
        conn.close()
        assert row is not None
        assert row[1] == "seed"

    def test_seed_handle_written_to_candidate_entities_not_venues(
        self, db_path: Path, tmp_path: Path, empty_blocklist: Path
    ) -> None:
        seeds = tmp_path / "seeds.yaml"
        seeds.write_text(
            yaml.dump({"handles": ["@thevaultlounge"], "venues": []})
        )

        _make_service(db_path, seeds, empty_blocklist).run()

        assert "@thevaultlounge" in _candidate_handles(db_path)
        assert _venue_count(db_path) == 0

    def test_seed_venue_address_is_geocoded(
        self, db_path: Path, tmp_path: Path, empty_blocklist: Path
    ) -> None:
        seeds = tmp_path / "seeds.yaml"
        seeds.write_text(
            yaml.dump(
                {
                    "handles": [],
                    "venues": [{"name": "Cinema Salem", "address": "95 Washington St"}],
                }
            )
        )
        geocoder = MagicMock(spec=GeocoderProvider)
        geocoder.geocode.return_value = (42.519, -70.897)

        _make_service(db_path, seeds, empty_blocklist, geocoder=geocoder).run()

        geocoder.geocode.assert_called_once_with("95 Washington St")

        conn = sqlite3.connect(db_path)
        row = conn.execute(
            "SELECT latitude, longitude FROM venues WHERE name='Cinema Salem'"
        ).fetchone()
        conn.close()
        assert row[0] == pytest.approx(42.519)
        assert row[1] == pytest.approx(-70.897)

    def test_geocoding_returns_none_stores_venue_with_null_coordinates(
        self, db_path: Path, tmp_path: Path, empty_blocklist: Path
    ) -> None:
        seeds = tmp_path / "seeds.yaml"
        seeds.write_text(
            yaml.dump(
                {
                    "handles": [],
                    "venues": [{"name": "Mystery Venue", "address": "1 Unknown Ln"}],
                }
            )
        )
        geocoder = MagicMock(spec=GeocoderProvider)
        geocoder.geocode.return_value = None

        warn_msgs: list[str] = []
        logger = MagicMock()
        logger.warning = lambda msg, **kw: warn_msgs.append(msg)

        _make_service(
            db_path, seeds, empty_blocklist, geocoder=geocoder, logger=logger
        ).run()

        conn = sqlite3.connect(db_path)
        row = conn.execute(
            "SELECT latitude, longitude FROM venues WHERE name='Mystery Venue'"
        ).fetchone()
        conn.close()
        assert row is not None
        assert row[0] is None
        assert row[1] is None
        assert any("geocod" in m.lower() for m in warn_msgs)

    def test_geocoding_exception_stores_venue_with_null_coordinates(
        self, db_path: Path, tmp_path: Path, empty_blocklist: Path
    ) -> None:
        seeds = tmp_path / "seeds.yaml"
        seeds.write_text(
            yaml.dump(
                {
                    "handles": [],
                    "venues": [{"name": "Crash Venue", "address": "0 Error Rd"}],
                }
            )
        )
        geocoder = MagicMock(spec=GeocoderProvider)
        geocoder.geocode.side_effect = RuntimeError("geocoder down")

        _make_service(db_path, seeds, empty_blocklist, geocoder=geocoder).run()

        conn = sqlite3.connect(db_path)
        row = conn.execute(
            "SELECT latitude, longitude FROM venues WHERE name='Crash Venue'"
        ).fetchone()
        conn.close()
        assert row is not None
        assert row[0] is None
        assert row[1] is None
