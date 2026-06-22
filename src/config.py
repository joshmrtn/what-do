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
class VenueDiscoveryConfig:
    categories: list[str] = field(
        default_factory=lambda: [
            "cafe", "theater", "music_venue", "bar",
            "restaurant", "museum", "park",
        ]
    )
    name_match_threshold: float = 0.92
    address_match_threshold: float = 0.85
    blocklist_name_match_threshold: float = 0.80


@dataclass
class DeduplicationConfig:
    fuzzy_title_threshold: float = 0.85
    time_window_hours: float = 2.0
    semantic_threshold: float = 0.92


@dataclass
class WeatherConfig:
    """Weather provider configuration."""

    provider: str = "open-meteo"


@dataclass
class ScoringConfig:
    """Scoring thresholds and multipliers."""

    top_picks_min: float = 0.5
    worth_considering_min: float = 0.1
    summary_weight: float = 0.3
    match_multiplier_yes: float = 1.5
    match_multiplier_maybe: float = 1.0
    match_multiplier_no: float = 0.5
    min_tags_per_event: int = 5


@dataclass
class SyntheticConditions:
    """Environmental conditions that must be satisfied to generate a synthetic activity."""

    min_temp_f: float | None = None
    max_temp_f: float | None = None
    weather: list[str] = field(default_factory=list)
    time_window: str | None = None


@dataclass
class SyntheticActivityRule:
    """A single rule for generating a synthetic activity event."""

    name: str
    conditions: SyntheticConditions
    tags: list[str]
    summary: str


@dataclass
class AppConfig:
    location: LocationConfig
    scraping: ScrapingConfig
    venue_discovery: VenueDiscoveryConfig
    deduplication: DeduplicationConfig = field(default_factory=DeduplicationConfig)
    weather: WeatherConfig = field(default_factory=WeatherConfig)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)
    synthetic_activities: list[SyntheticActivityRule] = field(default_factory=list)
    ollama_host: str = "http://localhost:11434"


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

    vd_data = data.get("venue_discovery", {})
    venue_discovery = VenueDiscoveryConfig(
        categories=vd_data.get(
            "categories",
            ["cafe", "theater", "music_venue", "bar", "restaurant", "museum", "park"],
        ),
        name_match_threshold=float(vd_data.get("name_match_threshold", 0.92)),
        address_match_threshold=float(vd_data.get("address_match_threshold", 0.85)),
        blocklist_name_match_threshold=float(
            vd_data.get("blocklist_name_match_threshold", 0.80)
        ),
    )

    dedup_data = data.get("deduplication", {})
    deduplication = DeduplicationConfig(
        fuzzy_title_threshold=float(dedup_data.get("fuzzy_title_threshold", 0.85)),
        time_window_hours=float(dedup_data.get("time_window_hours", 2.0)),
        semantic_threshold=float(dedup_data.get("semantic_threshold", 0.92)),
    )

    weather_data = data.get("weather", {})
    weather = WeatherConfig(
        provider=weather_data.get("provider", "open-meteo"),
    )

    scoring_data = data.get("scoring", {})
    tiers_data = scoring_data.get("tiers", {})
    multipliers_data = scoring_data.get("match_multipliers", {})
    scoring = ScoringConfig(
        top_picks_min=float(tiers_data.get("top_picks_min", 0.5)),
        worth_considering_min=float(tiers_data.get("worth_considering_min", 0.1)),
        summary_weight=float(scoring_data.get("summary_weight", 0.3)),
        match_multiplier_yes=float(multipliers_data.get("yes", 1.5)),
        match_multiplier_maybe=float(multipliers_data.get("maybe", 1.0)),
        match_multiplier_no=float(multipliers_data.get("no", 0.5)),
        min_tags_per_event=int(scoring_data.get("min_tags_per_event", 5)),
    )

    synthetic_activities: list[SyntheticActivityRule] = []
    for rule_data in data.get("synthetic_activities", []):
        cond_data = rule_data.get("conditions", {})
        conditions = SyntheticConditions(
            min_temp_f=float(cond_data["min_temp_f"]) if "min_temp_f" in cond_data else None,
            max_temp_f=float(cond_data["max_temp_f"]) if "max_temp_f" in cond_data else None,
            weather=list(cond_data.get("weather", [])),
            time_window=cond_data.get("time_window"),
        )
        synthetic_activities.append(
            SyntheticActivityRule(
                name=str(rule_data["name"]),
                conditions=conditions,
                tags=list(rule_data.get("tags", [])),
                summary=str(rule_data.get("summary", "")),
            )
        )

    return AppConfig(
        location=location,
        scraping=scraping,
        venue_discovery=venue_discovery,
        deduplication=deduplication,
        weather=weather,
        scoring=scoring,
        synthetic_activities=synthetic_activities,
        ollama_host=os.environ.get("OLLAMA_HOST", "http://localhost:11434"),
    )
