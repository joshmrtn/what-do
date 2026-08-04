from __future__ import annotations

import os
from typing import Any
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


@dataclass(frozen=True)
class ComfortCurve:
    """An asymmetric trapezoid mapping a weather reading to comfort in -1.0..+1.0.

    Comfort is +1.0 across the whole `ideal` band, ramps linearly to 0.0 at the
    `zero` bounds and on to -1.0 at the `floor` bounds, then clamps. A plateau
    rather than a peak, because a comfortable range has no single optimum; linear
    rather than logistic because a physical reading has no noise floor to crush.

    A side whose three bounds coincide is unbounded — readings past it stay ideal.

    Args:
        ideal: (low, high) bounds of the band scoring +1.0.
        zero: (low, high) readings scoring 0.0.
        floor: (low, high) readings scoring -1.0.
        weight: Share of the weighted mean. Ignored when `supersedes` is set.
        fallback_for: Another factor this one stands in for. Scored only when
            that factor has no reading, so correlated pairs cannot double-count.
        supersedes: Conditions whose categorical penalty this curve replaces.
            A curve with this set is a capping factor: it is excluded from the
            weighted mean and applied as an upper bound on total comfort, so
            intensity decides instead of a coarse condition label.
    """

    ideal: tuple[float, float]
    zero: tuple[float, float]
    floor: tuple[float, float]
    weight: float = 1.0
    fallback_for: str | None = None
    supersedes: tuple[str, ...] = ()


@dataclass
class WeatherConfig:
    """Weather provider and comfort scoring configuration."""

    provider: str = "open-meteo"
    #: Hour of the local day sampled for events with no known start time.
    default_hour: int = 20
    max_positive_adjustment: float = 0.15
    max_negative_adjustment: float = 0.25
    air_quality_enabled: bool = True
    #: Reading name -> curve. Names must match keys in the weather dict.
    comfort: dict[str, ComfortCurve] = field(default_factory=dict)
    #: Condition -> comfort ceiling. Only negative values cap.
    condition_penalty: dict[str, float] = field(default_factory=dict)


#: Contribution aggregation strategies. See docs/decisions.md — "Scoring formula
#: replaced after measurement" for why balanced_mean is the default.
AGGREGATORS = ("balanced_mean", "specificity_sum")

#: Domain applied to any source_type absent from scoring.domain_map.
GENERAL_DOMAIN_DEFAULT = "general"


@dataclass
class ScoringConfig:
    """Scoring thresholds, multipliers, and similarity shaping."""

    top_picks_min: float = 0.5
    worth_considering_min: float = 0.1
    summary_weight: float = 0.3
    match_multiplier_yes: float = 1.5
    match_multiplier_maybe: float = 1.0
    match_multiplier_no: float = 0.5
    min_tags_per_event: int = 5
    # Logistic gate. Measured with nomic-embed-text: unrelated pairs sit at
    # 0.33-0.48 and related pairs at 0.77-0.86, so 0.60 falls between them.
    gate_midpoint: float = 0.60
    gate_temperature: float = 0.04
    aggregator: str = "balanced_mean"
    # Relative-margin classification: an absolute dislike cutoff force-rejects
    # a karaoke bar, where 'bar' scores 0.932 against the dislike 'bars'.
    match_yes_min: float = 0.30
    match_no_margin: float = 0.15
    #: source_type -> preference domain. Unmapped types get [general] only.
    domain_map: dict[str, str] = field(default_factory=dict)


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
    gemini_api_key: str | None = None
    gemini_model: str = "gemini-flash-latest"


#: The only values `Event.setting` may take.
SETTINGS = ("indoor", "outdoor", "unknown")


def _bounds(raw: dict[str, Any], key: str, factor: str) -> tuple[float, float]:
    """Read one (low, high) bound pair from a comfort curve block."""
    if key not in raw:
        raise ConfigError(f"Comfort curve '{factor}' missing required bound: '{key}'")
    pair = raw[key]
    if not isinstance(pair, (list, tuple)) or len(pair) != 2:
        raise ConfigError(f"Comfort curve '{factor}' bound '{key}' must be [low, high]")
    low, high = float(pair[0]), float(pair[1])
    if low > high:
        raise ConfigError(f"Comfort curve '{factor}' bound '{key}' is inverted: {low} > {high}")
    return low, high


def _load_curve(factor: str, raw: dict[str, Any]) -> ComfortCurve:
    """Build one comfort curve, rejecting bounds that do not nest outward.

    A misordered bound silently inverts the curve's meaning, so it fails at load
    rather than quietly scoring a heatwave as pleasant.
    """
    ideal = _bounds(raw, "ideal", factor)
    zero = _bounds(raw, "zero", factor)
    floor = _bounds(raw, "floor", factor)

    if zero[0] > ideal[0] or zero[1] < ideal[1]:
        raise ConfigError(f"Comfort curve '{factor}': 'zero' must lie outside 'ideal'")
    if floor[0] > zero[0] or floor[1] < zero[1]:
        raise ConfigError(f"Comfort curve '{factor}': 'floor' must lie outside 'zero'")

    return ComfortCurve(
        ideal=ideal,
        zero=zero,
        floor=floor,
        weight=float(raw.get("weight", 1.0)),
        fallback_for=raw.get("fallback_for"),
        supersedes=tuple(raw.get("supersedes", ())),
    )


def _load_weather(raw: dict[str, Any]) -> WeatherConfig:
    """Build weather and comfort config, validating every tunable."""
    comfort = {
        factor: _load_curve(factor, curve_data)
        for factor, curve_data in (raw.get("comfort") or {}).items()
    }

    for factor, curve in comfort.items():
        if curve.fallback_for is not None and curve.fallback_for not in comfort:
            raise ConfigError(
                f"Comfort curve '{factor}' declares fallback_for "
                f"'{curve.fallback_for}', which is not a configured factor"
            )

    for name in ("max_positive_adjustment", "max_negative_adjustment"):
        if float(raw.get(name, 0.0)) < 0:
            raise ConfigError(f"Invalid {name}: must be non-negative")

    default_hour = int(raw.get("default_hour", 20))
    if not 0 <= default_hour <= 23:
        raise ConfigError(f"Invalid default_hour {default_hour}: must be between 0 and 23")

    defaults = WeatherConfig()
    return WeatherConfig(
        provider=raw.get("provider", defaults.provider),
        default_hour=default_hour,
        max_positive_adjustment=float(
            raw.get("max_positive_adjustment", defaults.max_positive_adjustment)
        ),
        max_negative_adjustment=float(
            raw.get("max_negative_adjustment", defaults.max_negative_adjustment)
        ),
        air_quality_enabled=bool(
            (raw.get("air_quality") or {}).get("enabled", defaults.air_quality_enabled)
        ),
        comfort=comfort,
        condition_penalty={
            str(k): float(v) for k, v in (raw.get("condition_penalty") or {}).items()
        },
    )


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

    weather = _load_weather(data.get("weather", {}))

    scoring_data = data.get("scoring", {})
    tiers_data = scoring_data.get("tiers", {})
    multipliers_data = scoring_data.get("match_multipliers", {})
    match_data = scoring_data.get("match", {})

    aggregator = str(scoring_data.get("aggregator", "balanced_mean"))
    if aggregator not in AGGREGATORS:
        raise ConfigError(
            f"Invalid scoring aggregator {aggregator!r}: must be one of {AGGREGATORS}"
        )

    # The ranking engine divides by the multiplier when the base score is
    # negative, so that a `no` deepens a bad score instead of improving it.
    # Zero would raise, and a negative multiplier would flip the sign.
    multipliers = {
        "yes": float(multipliers_data.get("yes", 1.5)),
        "maybe": float(multipliers_data.get("maybe", 1.0)),
        "no": float(multipliers_data.get("no", 0.5)),
    }
    for label, value in multipliers.items():
        if value <= 0:
            raise ConfigError(
                f"Invalid match multiplier for {label!r}: {value} — must be positive"
            )

    scoring = ScoringConfig(
        top_picks_min=float(tiers_data.get("top_picks_min", 0.5)),
        worth_considering_min=float(tiers_data.get("worth_considering_min", 0.1)),
        summary_weight=float(scoring_data.get("summary_weight", 0.3)),
        match_multiplier_yes=multipliers["yes"],
        match_multiplier_maybe=multipliers["maybe"],
        match_multiplier_no=multipliers["no"],
        min_tags_per_event=int(scoring_data.get("min_tags_per_event", 5)),
        gate_midpoint=float(scoring_data.get("gate_midpoint", 0.60)),
        gate_temperature=float(scoring_data.get("gate_temperature", 0.04)),
        aggregator=aggregator,
        match_yes_min=float(match_data.get("yes_min", 0.30)),
        match_no_margin=float(match_data.get("no_margin", 0.15)),
        domain_map={str(k): str(v) for k, v in scoring_data.get("domain_map", {}).items()},
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
        gemini_api_key=os.environ.get("GEMINI_API_KEY"),
        gemini_model=os.environ.get("GEMINI_MODEL", "gemini-flash-latest"),
    )
