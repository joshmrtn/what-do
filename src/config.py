from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import yaml
from dotenv import load_dotenv


class ConfigError(ValueError):
    """Raised when config.yaml is missing required fields or is malformed."""


@dataclass
class LocationConfig:
    latitude: float
    longitude: float
    postal_code: str
    search_radius_miles: float


@dataclass
class AppConfig:
    location: LocationConfig
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
        ConfigError: If required config fields are missing.
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
    for field in ("latitude", "longitude", "postal_code", "search_radius_miles"):
        if field not in loc:
            raise ConfigError(f"Config missing required location field: '{field}'")

    location = LocationConfig(
        latitude=float(loc["latitude"]),
        longitude=float(loc["longitude"]),
        postal_code=str(loc["postal_code"]),
        search_radius_miles=float(loc["search_radius_miles"]),
    )

    return AppConfig(
        location=location,
        ollama_host=os.environ.get("OLLAMA_HOST", "http://localhost:11434"),
    )
