from __future__ import annotations

import os
from typing import Any
from dataclasses import dataclass, field
from datetime import time
from functools import lru_cache
from pathlib import Path

import yaml
from dotenv import load_dotenv
from timezonefinder import TimezoneFinder

from src.models.source_type import RESERVED


class ConfigError(ValueError):
    """Raised when config.yaml is missing required fields or is malformed."""


@lru_cache(maxsize=1)
def _timezone_finder() -> TimezoneFinder:
    """Return the process-wide TimezoneFinder, building it on first use.

    Constructing one loads its bundled boundary data and measures ~0.68s, while
    the lookup itself is ~0.004s and the data is static. Building it per call
    made `load_config` cost more than everything it parses.
    """
    return TimezoneFinder()


@dataclass
class LocationConfig:
    latitude: float
    longitude: float
    postal_code: str
    search_radius_miles: float
    timezone: str


#: How many whole nights ahead an event may start and still be ingested and
#: ranked. Measured: northshorenightout publishes ~39 days out, so the earlier
#: 30 truncated it. Adapters import this for their constructor fallbacks, so the
#: default cannot drift between `config.yaml` and a call site.
DEFAULT_HORIZON_DAYS = 45


@dataclass
class ScrapingConfig:
    lookback_days: int = 30
    #: How far ahead of the run date an event may start and still be ranked.
    #: There is no matching bound on how long ago it was *announced*.
    horizon_days: int = DEFAULT_HORIZON_DAYS
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
class FeedConfig:
    """One fetched event source, declared in config rather than code.

    Attributes:
        name: Identifier for the source, used as its `source` value.
        url: Address of the feed or listing page.
        source_type: Provenance label on every candidate. Defaults to `name`.
        min_fetch_interval_hours: Politeness floor. Re-running the batch by hand
            within this window reuses the cached copy rather than refetching.
        venue: Venue every event in this feed belongs to, for single-venue
            sources whose entries do not name it. Filled in only where the feed
            itself declares none — a summary that names a venue is more specific
            and wins.
        city: City for the same, on the same terms.
        site_url: Human-facing page to attribute this source's events to when an
            event carries no URL of its own. Declared separately because `url` is
            whatever we fetch, which is often not something to hand a person — a
            Google Calendar ICS export is a correct feed and a useless link.
    """

    name: str
    url: str
    source_type: str
    min_fetch_interval_hours: float = 6.0
    venue: str | None = None
    city: str | None = None
    site_url: str | None = None


@dataclass
class SourcesConfig:
    """Declared event sources that are configured rather than coded.

    The two lists differ only in how the fetched document is parsed. A site can
    legitimately appear in both — a calendar feed and a listing page often cover
    different slices of the same venues.
    """

    ics_calendars: list[FeedConfig] = field(default_factory=list)
    html_calendars: list[FeedConfig] = field(default_factory=list)
    #: Veezi public ticketing pages. One entry per cinema; the siteToken that
    #: identifies it is part of the URL, so no code changes to add another.
    veezi_cinemas: list[FeedConfig] = field(default_factory=list)
    #: Paginated `whats-on` listings in The Cabot's markup.
    cabot_listings: list[FeedConfig] = field(default_factory=list)
    #: The Events Calendar (WordPress) iCal exports, whose 30-event cap is
    #: walked with `tribe-bar-date` rather than paged.
    tribe_calendars: list[FeedConfig] = field(default_factory=list)
    #: Do617 venue pages, whose schema.org microdata states each event's own
    #: UTC offset. One entry per venue worth watching.
    do617_venues: list[FeedConfig] = field(default_factory=list)
    #: MOON's Squarespace show feed, whose event dates live in item titles.
    moon_feeds: list[FeedConfig] = field(default_factory=list)
    #: Assabet Interactive calendar feeds, whose pubDate *is* the event start.
    assabet_feeds: list[FeedConfig] = field(default_factory=list)
    #: Pages publishing schema.org events as JSON-LD — the richest markup a
    #: site can offer, stating offsets and cancellations outright.
    jsonld_pages: list[FeedConfig] = field(default_factory=list)

    def all_feeds(self) -> list[FeedConfig]:
        """Every declared feed, across all parser groups, in declaration order."""
        feeds: list[FeedConfig] = []
        for value in vars(self).values():
            if isinstance(value, list):
                feeds.extend(value)
        return feeds

    def site_url_by_source_type(self) -> dict[str, str]:
        """Human-facing page per source_type, for events carrying no URL.

        Only unambiguous source types get an entry. Several feeds may share a
        `source_type` — two Veezi cinemas both produce `cinema_veezi` — and an
        event records only that label, so where the feeds disagree there is no
        way to tell which site an event came from. A wrong link is worse than
        none, so a disagreement yields no entry at all.
        """
        candidates: dict[str, set[str]] = {}
        for feed in self.all_feeds():
            candidates.setdefault(feed.source_type, set()).add(feed.site_url or feed.url)

        return {
            source_type: next(iter(urls))
            for source_type, urls in candidates.items()
            if len(urls) == 1
        }


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
    #: How long a cached forecast may be served. Under 24h so a nightly batch
    #: always rescores against a forecast issued that night, never one issued
    #: days earlier when the event was first discovered.
    cache_ttl_hours: float = 12.0
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
    #: Synthetic activities bypass LLM extraction, so the rule author is the only
    #: source of indoor/outdoor. Only they know if a new rule happens indoors.
    setting: str = "unknown"


#: Default model names. Providers import these for their constructor fallbacks, so
#: a default cannot drift between `config.yaml` and the call site.
DEFAULT_EXTRACTION_MODEL = "gemma4:e4b"
DEFAULT_DISAMBIGUATION_MODEL = "gemma4:e2b"
DEFAULT_EMBEDDING_MODEL = "nomic-embed-text"


#: Ceiling on one model call, in seconds. Twenty minutes, because the batch VM
#: runs these on four CPU cores with no GPU.
DEFAULT_LLM_TIMEOUT_SECONDS = 1200

#: Ollama's own default is 4096, and a model that reasons fills that with
#: thinking before a single character of the answer is emitted — which is how
#: 44% of one night's extractions came back with empty content. 32768 leaves
#: room to reason and still answer. Past ~64k the model is not thinking, it is
#: looping, and a bigger window buys nothing.
DEFAULT_LLM_NUM_CTX = 32768

#: gemma4:e4b ships temperature 1 / top_p 0.95, which is sampling tuned for
#: variety. Extraction wants the opposite: the same JSON shape every time.
DEFAULT_LLM_TEMPERATURE = 0.2
DEFAULT_LLM_TOP_P = 0.9

#: How long a model stays resident after a call. Long enough that the model
#: stays hot across a stage whose calls are minutes apart, short enough that it
#: releases its footprint once the batch is done. Explicit because this host's
#: server default never expires: a model loaded once by a stray test pins its
#: whole footprint until Ollama restarts.
DEFAULT_LLM_KEEP_ALIVE = "30m"


@dataclass
class ModelsConfig:
    """Model names and request parameters for the model-backed stages."""

    llm_extraction: str = DEFAULT_EXTRACTION_MODEL
    llm_disambiguation: str = DEFAULT_DISAMBIGUATION_MODEL
    embeddings: str = DEFAULT_EMBEDDING_MODEL
    #: How long one model call may take. Generous because the target is CPU
    #: inference: measured on the batch VM, a single extraction runs for
    #: minutes, so the old 60-second default failed *every* call and left the
    #: whole batch with no tags at all.
    request_timeout_seconds: int = DEFAULT_LLM_TIMEOUT_SECONDS
    temperature: float = DEFAULT_LLM_TEMPERATURE
    top_p: float = DEFAULT_LLM_TOP_P
    num_ctx: int = DEFAULT_LLM_NUM_CTX
    #: Asks a thinking-capable model to answer directly. Advisory: some models
    #: reason anyway, which is why `num_ctx` is sized to survive being ignored.
    think: bool = False
    #: Ollama's constrained decoding. "json" makes unparseable output
    #: structurally impossible rather than merely discouraged by the prompt.
    #: None sends nothing and lets the model emit whatever it likes.
    response_format: str | None = "json"
    #: None defers to the server's own default.
    keep_alive: str | None = DEFAULT_LLM_KEEP_ALIVE


#: When one night's listing gives way to the next. Not midnight: a calendar
#: rollover empties the evening still in progress, and someone asking at 00:30
#: means the night they are standing in.
DEFAULT_DAY_STARTS_AT = time(4, 0)


@dataclass
class AppConfig:
    location: LocationConfig
    scraping: ScrapingConfig
    venue_discovery: VenueDiscoveryConfig
    deduplication: DeduplicationConfig = field(default_factory=DeduplicationConfig)
    weather: WeatherConfig = field(default_factory=WeatherConfig)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)
    sources: SourcesConfig = field(default_factory=SourcesConfig)
    models: ModelsConfig = field(default_factory=ModelsConfig)
    day_starts_at: time = DEFAULT_DAY_STARTS_AT
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

    cache_ttl_hours = float(raw.get("cache_ttl_hours", defaults.cache_ttl_hours))
    if cache_ttl_hours <= 0:
        raise ConfigError(
            f"Invalid cache_ttl_hours {cache_ttl_hours}: must be positive"
        )

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
        cache_ttl_hours=cache_ttl_hours,
        comfort=comfort,
        condition_penalty={
            str(k): float(v) for k, v in (raw.get("condition_penalty") or {}).items()
        },
    )


def _optional_text(entry: dict[str, Any], key: str, kind: str, name: str) -> str | None:
    """Read an optional string, rejecting a blank rather than accepting it.

    A blank is a typo, not a choice: it reads as "declared" while attributing
    nothing, which is harder to notice than leaving the key out.
    """
    if key not in entry or entry[key] is None:
        return None

    value = str(entry[key]).strip()
    if not value:
        raise ConfigError(f"{kind} '{name}' has a blank {key}")

    return value


def _load_feeds(entries: Any, kind: str) -> list[FeedConfig]:
    """Build one list of fetched sources, rejecting entries that cannot be used.

    Validation is strict on purpose: these run unattended overnight, so a typo
    should fail at load rather than silently ingest nothing at 2am.
    """
    feeds = []

    for index, entry in enumerate(entries or []):
        for required in ("name", "url"):
            if not entry.get(required):
                raise ConfigError(
                    f"{kind} at position {index} missing required field: '{required}'"
                )

        name = str(entry["name"])
        interval = float(entry.get("min_fetch_interval_hours", 6.0))
        if interval < 0:
            raise ConfigError(
                f"{kind} '{name}' has a negative "
                f"min_fetch_interval_hours: {interval}"
            )

        venue = _optional_text(entry, "venue", kind, name)
        city = _optional_text(entry, "city", kind, name)
        site_url = _optional_text(entry, "site_url", kind, name)

        source_type = str(entry.get("source_type", name))
        if source_type in RESERVED:
            raise ConfigError(
                f"{kind} '{name}' claims reserved source_type '{source_type}'. "
                "Reserved values change how a stage treats an event, so a feed "
                "would silently inherit that behaviour."
            )

        feeds.append(
            FeedConfig(
                name=name,
                url=str(entry["url"]),
                source_type=source_type,
                min_fetch_interval_hours=interval,
                venue=venue,
                city=city,
                site_url=site_url,
            )
        )

    return feeds


def _load_day_starts_at(raw: dict[str, Any]) -> time:
    """Read the rollover hour, rejecting an unparseable value.

    A malformed value is not defaulted away: `day_starts_at` decides which day
    the whole system believes it is in, so a typo has to fail loudly rather
    than silently shifting ingestion and the CLI onto a different night.
    """
    if "day_starts_at" not in raw:
        return DEFAULT_DAY_STARTS_AT

    value = raw["day_starts_at"]
    if isinstance(value, time):
        return value

    # YAML 1.1 reads an unquoted `4:00` as sexagesimal and hands back the int
    # 240, so the quoting advice is the actual fix far more often than the
    # format is. `04:00` happens to survive unquoted, which makes the trap
    # intermittent rather than obvious.
    complaint = f"Invalid day_starts_at {value!r}: expected HH:MM, quoted"

    parts = str(value).split(":")
    if not 2 <= len(parts) <= 3 or not all(p.isdigit() for p in parts):
        raise ConfigError(complaint)

    try:
        parsed = time.fromisoformat(str(value))
    except ValueError as exc:
        raise ConfigError(complaint) from exc

    return parsed


def _load_sources(raw: dict[str, Any]) -> SourcesConfig:
    """Build every configured source list."""
    return SourcesConfig(
        ics_calendars=_load_feeds(raw.get("ics_calendars"), "ICS calendar"),
        html_calendars=_load_feeds(raw.get("html_calendars"), "HTML calendar"),
        veezi_cinemas=_load_feeds(raw.get("veezi_cinemas"), "Veezi cinema"),
        cabot_listings=_load_feeds(raw.get("cabot_listings"), "Cabot listing"),
        tribe_calendars=_load_feeds(raw.get("tribe_calendars"), "Tribe calendar"),
        do617_venues=_load_feeds(raw.get("do617_venues"), "Do617 venue"),
        moon_feeds=_load_feeds(raw.get("moon_feeds"), "MOON feed"),
        assabet_feeds=_load_feeds(raw.get("assabet_feeds"), "Assabet feed"),
        jsonld_pages=_load_feeds(raw.get("jsonld_pages"), "JSON-LD page"),
    )


def _load_models(raw: dict[str, Any]) -> ModelsConfig:
    """Build the model names, rejecting a blank name rather than calling with none."""
    defaults = ModelsConfig()

    def _name(key: str, default: str) -> str:
        if key not in raw:
            return default
        value = str(raw[key]).strip()
        if not value:
            raise ConfigError(f"Model name '{key}' is blank")
        return value

    timeout = defaults.request_timeout_seconds
    if "request_timeout_seconds" in raw:
        try:
            timeout = int(raw["request_timeout_seconds"])
        except (TypeError, ValueError) as error:
            raise ConfigError(
                f"models.request_timeout_seconds is not a number: {raw['request_timeout_seconds']!r}"
            ) from error
        if timeout <= 0:
            raise ConfigError(
                f"models.request_timeout_seconds must be positive, got {timeout}"
            )

    def _number(key: str, default: float, low: float, high: float | None) -> float:
        """Read a numeric parameter, rejecting anything outside its range."""
        if key not in raw:
            return default
        try:
            value = float(raw[key])
        except (TypeError, ValueError) as error:
            raise ConfigError(f"models.{key} is not a number: {raw[key]!r}") from error
        if value < low or (high is not None and value > high):
            bound = f"{low}..{high}" if high is not None else f"at least {low}"
            raise ConfigError(f"models.{key} must be {bound}, got {value}")
        return value

    num_ctx = int(_number("num_ctx", float(defaults.num_ctx), 1, None))
    temperature = _number("temperature", defaults.temperature, 0.0, None)
    # Excludes zero: top_p 0 admits no tokens at all.
    top_p = _number("top_p", defaults.top_p, 0.0, 1.0)
    if "top_p" in raw and top_p == 0:
        raise ConfigError("models.top_p must be greater than 0")

    # A blank format is how config turns constrained decoding off, distinct
    # from omitting the key, which keeps the default on.
    response_format = defaults.response_format
    if "format" in raw:
        response_format = str(raw["format"]).strip() or None

    keep_alive = defaults.keep_alive
    if "keep_alive" in raw:
        keep_alive = str(raw["keep_alive"]).strip() or None

    return ModelsConfig(
        llm_extraction=_name("llm_extraction", defaults.llm_extraction),
        llm_disambiguation=_name("llm_disambiguation", defaults.llm_disambiguation),
        embeddings=_name("embeddings", defaults.embeddings),
        request_timeout_seconds=timeout,
        temperature=temperature,
        top_p=top_p,
        num_ctx=num_ctx,
        think=bool(raw.get("think", defaults.think)),
        response_format=response_format,
        keep_alive=keep_alive,
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

    tz_name = _timezone_finder().timezone_at(lat=latitude, lng=longitude)
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
    horizon_days = int(scraping_data.get("horizon_days", DEFAULT_HORIZON_DAYS))
    if horizon_days <= 0:
        raise ConfigError(f"Invalid horizon_days: {horizon_days} — must be positive")

    scraping = ScrapingConfig(
        lookback_days=int(scraping_data.get("lookback_days", 30)),
        horizon_days=horizon_days,
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
        name = str(rule_data["name"])
        setting = str(rule_data.get("setting", "unknown"))
        if setting not in SETTINGS:
            raise ConfigError(
                f"Synthetic activity '{name}' has setting '{setting}', "
                f"outside the allowed values {SETTINGS}"
            )
        synthetic_activities.append(
            SyntheticActivityRule(
                name=name,
                conditions=conditions,
                tags=list(rule_data.get("tags", [])),
                summary=str(rule_data.get("summary", "")),
                setting=setting,
            )
        )

    return AppConfig(
        location=location,
        scraping=scraping,
        venue_discovery=venue_discovery,
        deduplication=deduplication,
        weather=weather,
        scoring=scoring,
        sources=_load_sources(data.get("sources") or {}),
        models=_load_models(data.get("models") or {}),
        day_starts_at=_load_day_starts_at(data),
        synthetic_activities=synthetic_activities,
        ollama_host=os.environ.get("OLLAMA_HOST", "http://localhost:11434"),
        gemini_api_key=os.environ.get("GEMINI_API_KEY"),
        gemini_model=os.environ.get("GEMINI_MODEL", "gemini-flash-latest"),
    )
