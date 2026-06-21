from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from dotenv import load_dotenv
from timezonefinder import TimezoneFinder


class ConfigError(ValueError):
    """Raised when config.yaml is missing required fields or is malformed."""


@dataclass
class LocationConfig:
    latitude: float
    longitude: float
    postal_code: str
    search_radius_miles: float
    timezone: str


@dataclass
class ScrapingConfig:
    lookback_days: int = 30
    max_discovery_depth: int = 2
    candidate_promotion_threshold: int = 3


@dataclass
class AppConfig:
    location: LocationConfig
    scraping: ScrapingConfig
    ollama_host: str


def load_config(
    config_path: Path | str | None = None,
    env_path: Path | str | None = None,
) -> AppConfig:
    """Load and validate application config from YAML and environment.

    Args:
        config_path: Path to config.yaml. Defaults to config/config.yaml.
        env_path: Path to .env file. Defaults to .env in cwd.

    Returns:
        Validated AppConfig instance.

    Raises:
        ConfigError: If required config fields are missing or values are invalid.
    """
    if env_path is not None:
        load_dotenv(env_path)
    else:
        load_dotenv()

    if config_path is None:
        config_path = Path("config/config.yaml")

    with open(config_path) as f:
        data = yaml.safe_load(f) or {}

    if "location" not in data:
        raise ConfigError("Config missing required section: 'location'")

    loc = data["location"]
    for required in ("latitude", "longitude", "postal_code", "search_radius_miles"):
        if required not in loc:
            raise ConfigError(f"Config missing required location field: '{required}'")

    latitude = float(loc["latitude"])
    longitude = float(loc["longitude"])
    search_radius = float(loc["search_radius_miles"])

    if not -90 <= latitude <= 90:
        raise ConfigError(f"Invalid latitude {latitude}: must be between -90 and 90")
    if not -180 <= longitude <= 180:
        raise ConfigError(f"Invalid longitude {longitude}: must be between -180 and 180")
    if search_radius <= 0:
        raise ConfigError(f"Invalid search_radius_miles {search_radius}: must be positive")

    tz_name = TimezoneFinder().timezone_at(lat=latitude, lng=longitude)
    if tz_name is None:
        raise ConfigError(
            f"Could not derive timezone from coordinates ({latitude}, {longitude})"
        )

    location = LocationConfig(
        latitude=latitude,
        longitude=longitude,
        postal_code=str(loc["postal_code"]),
        search_radius_miles=search_radius,
        timezone=tz_name,
    )

    scraping_data = data.get("scraping", {})
    scraping = ScrapingConfig(
        lookback_days=int(scraping_data.get("lookback_days", 30)),
        max_discovery_depth=int(scraping_data.get("max_discovery_depth", 2)),
        candidate_promotion_threshold=int(
            scraping_data.get("candidate_promotion_threshold", 3)
        ),
    )

    return AppConfig(
        location=location,
        scraping=scraping,
        ollama_host=os.environ.get("OLLAMA_HOST", "http://localhost:11434"),
    )
