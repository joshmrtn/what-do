import os
from datetime import time

import pytest
import yaml

from src.config import (
    DEFAULT_HORIZON_DAYS,
    ConfigError,
    FeedConfig,
    SourcesConfig,
    _timezone_finder,
    load_config,
)


def _write_config(tmp_path, data):
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml.dump(data))
    return config_file


def _valid_location_data():
    return {
        "location": {
            "latitude": 42.52,
            "longitude": -70.89,
            "postal_code": "01970",
            "search_radius_miles": 10,
        }
    }


def _load(tmp_path, extra):
    """Load a config built from a valid location block plus `extra` sections."""
    data = _valid_location_data()
    data.update(extra)
    return load_config(config_path=_write_config(tmp_path, data))


def test_valid_config_loads(tmp_path):
    cfg = load_config(config_path=_write_config(tmp_path, _valid_location_data()))
    assert cfg.location.latitude == 42.52
    assert cfg.location.longitude == -70.89
    assert cfg.location.postal_code == "01970"
    assert cfg.location.search_radius_miles == 10


def test_missing_location_section_raises(tmp_path):
    with pytest.raises(ConfigError, match="location"):
        load_config(config_path=_write_config(tmp_path, {}))


def test_missing_latitude_raises(tmp_path):
    data = _valid_location_data()
    del data["location"]["latitude"]
    with pytest.raises(ConfigError, match="latitude"):
        load_config(config_path=_write_config(tmp_path, data))


def test_missing_longitude_raises(tmp_path):
    data = _valid_location_data()
    del data["location"]["longitude"]
    with pytest.raises(ConfigError, match="longitude"):
        load_config(config_path=_write_config(tmp_path, data))


def test_ollama_host_defaults_when_not_set(tmp_path, monkeypatch):
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    cfg = load_config(config_path=_write_config(tmp_path, _valid_location_data()))
    assert cfg.ollama_host == "http://localhost:11434"


def test_ollama_host_reads_from_env(tmp_path, monkeypatch):
    monkeypatch.setenv("OLLAMA_HOST", "http://gpu-box:11434")
    cfg = load_config(config_path=_write_config(tmp_path, _valid_location_data()))
    assert cfg.ollama_host == "http://gpu-box:11434"


def test_gemini_api_key_reads_from_env(tmp_path, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "secret-abc")
    cfg = load_config(config_path=_write_config(tmp_path, _valid_location_data()))
    assert cfg.gemini_api_key == "secret-abc"


def test_gemini_api_key_none_when_not_set(tmp_path, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    empty_env = tmp_path / ".env"
    empty_env.write_text("")
    cfg = load_config(
        config_path=_write_config(tmp_path, _valid_location_data()),
        env_path=empty_env,
    )
    assert cfg.gemini_api_key is None


def test_gemini_model_defaults_to_flash(tmp_path, monkeypatch):
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    cfg = load_config(config_path=_write_config(tmp_path, _valid_location_data()))
    assert cfg.gemini_model == "gemini-flash-latest"


def test_gemini_model_reads_from_env(tmp_path, monkeypatch):
    monkeypatch.setenv("GEMINI_MODEL", "gemini-2.5-flash")
    cfg = load_config(config_path=_write_config(tmp_path, _valid_location_data()))
    assert cfg.gemini_model == "gemini-2.5-flash"


def test_dotenv_values_loaded(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("APIFY_API_KEY=test_key_abc\n")
    monkeypatch.delenv("APIFY_API_KEY", raising=False)
    load_config(
        config_path=_write_config(tmp_path, _valid_location_data()),
        env_path=env_file,
    )
    assert os.environ.get("APIFY_API_KEY") == "test_key_abc"


def test_missing_optional_secrets_no_error(tmp_path, monkeypatch):
    for key in ("APIFY_API_KEY", "TMDB_API_KEY", "AMC_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    load_config(config_path=_write_config(tmp_path, _valid_location_data()))


# --- bounds validation ---

def test_latitude_above_90_raises(tmp_path):
    data = _valid_location_data()
    data["location"]["latitude"] = 91.0
    with pytest.raises(ConfigError, match="latitude"):
        load_config(config_path=_write_config(tmp_path, data))


def test_latitude_below_neg90_raises(tmp_path):
    data = _valid_location_data()
    data["location"]["latitude"] = -91.0
    with pytest.raises(ConfigError, match="latitude"):
        load_config(config_path=_write_config(tmp_path, data))


def test_longitude_above_180_raises(tmp_path):
    data = _valid_location_data()
    data["location"]["longitude"] = 181.0
    with pytest.raises(ConfigError, match="longitude"):
        load_config(config_path=_write_config(tmp_path, data))


def test_longitude_below_neg180_raises(tmp_path):
    data = _valid_location_data()
    data["location"]["longitude"] = -181.0
    with pytest.raises(ConfigError, match="longitude"):
        load_config(config_path=_write_config(tmp_path, data))


def test_search_radius_zero_raises(tmp_path):
    data = _valid_location_data()
    data["location"]["search_radius_miles"] = 0
    with pytest.raises(ConfigError, match="search_radius_miles"):
        load_config(config_path=_write_config(tmp_path, data))


def test_search_radius_negative_raises(tmp_path):
    data = _valid_location_data()
    data["location"]["search_radius_miles"] = -5
    with pytest.raises(ConfigError, match="search_radius_miles"):
        load_config(config_path=_write_config(tmp_path, data))


# --- timezone derivation ---

def test_timezone_derived_from_coordinates(tmp_path):
    # 42.52, -70.89 is Salem MA → America/New_York
    cfg = load_config(config_path=_write_config(tmp_path, _valid_location_data()))
    assert cfg.location.timezone == "America/New_York"


# --- scraping defaults ---

def test_lookback_days_defaults_to_30(tmp_path):
    cfg = load_config(config_path=_write_config(tmp_path, _valid_location_data()))
    assert cfg.scraping.lookback_days == 30


def test_lookback_days_reads_from_config(tmp_path):
    data = _valid_location_data()
    data["scraping"] = {"lookback_days": 14}
    cfg = load_config(config_path=_write_config(tmp_path, data))
    assert cfg.scraping.lookback_days == 14


# --- deduplication config ---

def test_deduplication_defaults_when_section_absent(tmp_path):
    cfg = load_config(config_path=_write_config(tmp_path, _valid_location_data()))
    assert cfg.deduplication.fuzzy_title_threshold == 0.85
    assert cfg.deduplication.time_window_hours == 2.0
    assert cfg.deduplication.semantic_threshold == 0.92


def test_deduplication_reads_from_config(tmp_path):
    data = _valid_location_data()
    data["deduplication"] = {
        "fuzzy_title_threshold": 0.90,
        "time_window_hours": 4.0,
        "semantic_threshold": 0.95,
    }
    cfg = load_config(config_path=_write_config(tmp_path, data))
    assert cfg.deduplication.fuzzy_title_threshold == 0.90
    assert cfg.deduplication.time_window_hours == 4.0
    assert cfg.deduplication.semantic_threshold == 0.95


def test_deduplication_partial_overrides_keep_defaults(tmp_path):
    data = _valid_location_data()
    data["deduplication"] = {"fuzzy_title_threshold": 0.80}
    cfg = load_config(config_path=_write_config(tmp_path, data))
    assert cfg.deduplication.fuzzy_title_threshold == 0.80
    assert cfg.deduplication.time_window_hours == 2.0
    assert cfg.deduplication.semantic_threshold == 0.92


# --- weather config ---

def test_weather_section_absent_uses_defaults(tmp_path):
    cfg = load_config(config_path=_write_config(tmp_path, _valid_location_data()))
    assert cfg.weather.provider == "open-meteo"


def test_weather_provider_reads_from_config(tmp_path):
    data = _valid_location_data()
    data["weather"] = {"provider": "custom-provider"}
    cfg = load_config(config_path=_write_config(tmp_path, data))
    assert cfg.weather.provider == "custom-provider"


# --- scoring config ---

def test_scoring_section_absent_uses_defaults(tmp_path):
    cfg = load_config(config_path=_write_config(tmp_path, _valid_location_data()))
    assert cfg.scoring.summary_weight == 0.3
    assert cfg.scoring.match_multiplier_yes == 1.5
    assert cfg.scoring.match_multiplier_maybe == 1.0
    assert cfg.scoring.match_multiplier_no == 0.5
    assert cfg.scoring.min_tags_per_event == 5


def test_scoring_reads_from_config(tmp_path):
    data = _valid_location_data()
    data["scoring"] = {
        "summary_weight": 0.4,
        "match_multipliers": {"yes": 2.0, "maybe": 1.0, "no": 0.25},
        "min_tags_per_event": 8,
    }
    cfg = load_config(config_path=_write_config(tmp_path, data))
    assert cfg.scoring.summary_weight == 0.4
    assert cfg.scoring.match_multiplier_yes == 2.0
    assert cfg.scoring.match_multiplier_no == 0.25
    assert cfg.scoring.min_tags_per_event == 8


# --- synthetic activities config ---

def test_synthetic_activities_absent_returns_empty_list(tmp_path):
    cfg = load_config(config_path=_write_config(tmp_path, _valid_location_data()))
    assert cfg.synthetic_activities == []


def test_synthetic_activity_rule_parsed_correctly(tmp_path):
    data = _valid_location_data()
    data["synthetic_activities"] = [
        {
            "name": "Evening walk",
            "conditions": {
                "min_temp_f": 45.0,
                "max_temp_f": 85.0,
                "weather": ["clear", "partly_cloudy"],
            },
            "tags": ["outdoor", "walking", "low_key"],
            "summary": "A pleasant walk around town",
        }
    ]
    cfg = load_config(config_path=_write_config(tmp_path, data))
    assert len(cfg.synthetic_activities) == 1
    rule = cfg.synthetic_activities[0]
    assert rule.name == "Evening walk"
    assert rule.conditions.min_temp_f == 45.0
    assert rule.conditions.max_temp_f == 85.0
    assert rule.conditions.weather == ["clear", "partly_cloudy"]
    assert rule.conditions.time_window is None
    assert rule.tags == ["outdoor", "walking", "low_key"]
    assert rule.summary == "A pleasant walk around town"


def test_synthetic_activity_with_time_window(tmp_path):
    data = _valid_location_data()
    data["synthetic_activities"] = [
        {
            "name": "Sunset picnic",
            "conditions": {
                "min_temp_f": 65.0,
                "weather": ["clear"],
                "time_window": "sunset_minus_2h to sunset_plus_30min",
            },
            "tags": ["outdoor", "picnic"],
            "summary": "A picnic at sunset",
        }
    ]
    cfg = load_config(config_path=_write_config(tmp_path, data))
    rule = cfg.synthetic_activities[0]
    assert rule.conditions.time_window == "sunset_minus_2h to sunset_plus_30min"


def test_synthetic_activity_no_temp_constraints(tmp_path):
    data = _valid_location_data()
    data["synthetic_activities"] = [
        {
            "name": "Any time walk",
            "conditions": {},
            "tags": ["outdoor"],
            "summary": "A walk",
        }
    ]
    cfg = load_config(config_path=_write_config(tmp_path, data))
    rule = cfg.synthetic_activities[0]
    assert rule.conditions.min_temp_f is None
    assert rule.conditions.max_temp_f is None
    assert rule.conditions.weather == []
    assert rule.conditions.time_window is None


def test_multiple_synthetic_activity_rules(tmp_path):
    data = _valid_location_data()
    data["synthetic_activities"] = [
        {"name": "Walk", "conditions": {}, "tags": ["outdoor"], "summary": "Walk"},
        {"name": "Picnic", "conditions": {}, "tags": ["outdoor", "picnic"], "summary": "Picnic"},
    ]
    cfg = load_config(config_path=_write_config(tmp_path, data))
    assert len(cfg.synthetic_activities) == 2
    assert cfg.synthetic_activities[0].name == "Walk"
    assert cfg.synthetic_activities[1].name == "Picnic"


# ---------------------------------------------------------------------------
# Scoring: gate, aggregator, match thresholds, domain map
# ---------------------------------------------------------------------------


def test_scoring_gate_defaults(tmp_path):
    cfg = _load(tmp_path, {})

    assert cfg.scoring.gate_midpoint == 0.60
    assert cfg.scoring.gate_temperature == 0.04


def test_scoring_gate_overridable(tmp_path):
    cfg = _load(tmp_path, {"scoring": {"gate_midpoint": 0.55, "gate_temperature": 0.10}})

    assert cfg.scoring.gate_midpoint == 0.55
    assert cfg.scoring.gate_temperature == 0.10


def test_aggregator_defaults_to_balanced_mean(tmp_path):
    assert _load(tmp_path, {}).scoring.aggregator == "balanced_mean"


def test_aggregator_overridable(tmp_path):
    cfg = _load(tmp_path, {"scoring": {"aggregator": "specificity_sum"}})

    assert cfg.scoring.aggregator == "specificity_sum"


def test_unknown_aggregator_rejected(tmp_path):

    with pytest.raises(ConfigError, match="aggregator"):
        _load(tmp_path, {"scoring": {"aggregator": "vibes"}})


def test_match_thresholds_default(tmp_path):
    cfg = _load(tmp_path, {})

    assert cfg.scoring.match_yes_min == 0.30
    assert cfg.scoring.match_no_margin == 0.15


def test_match_thresholds_overridable(tmp_path):
    cfg = _load(tmp_path, {"scoring": {"match": {"yes_min": 0.5, "no_margin": 0.25}}})

    assert cfg.scoring.match_yes_min == 0.5
    assert cfg.scoring.match_no_margin == 0.25


def test_domain_map_defaults_empty(tmp_path):
    assert _load(tmp_path, {}).scoring.domain_map == {}


def test_domain_map_loaded_from_config(tmp_path):
    cfg = _load(tmp_path, {"scoring": {"domain_map": {"cinema_veezi": "movies", "amc": "movies"}}})

    assert cfg.scoring.domain_map == {"cinema_veezi": "movies", "amc": "movies"}


@pytest.mark.parametrize("label", ["yes", "maybe", "no"])
def test_zero_match_multiplier_rejected(tmp_path, label):
    """A negative base score divides by the multiplier, so zero would crash."""

    with pytest.raises(ConfigError, match="multiplier"):
        _load(tmp_path, {"scoring": {"match_multipliers": {label: 0}}})


@pytest.mark.parametrize("label", ["yes", "maybe", "no"])
def test_negative_match_multiplier_rejected(tmp_path, label):

    with pytest.raises(ConfigError, match="multiplier"):
        _load(tmp_path, {"scoring": {"match_multipliers": {label: -1.5}}})


def test_positive_match_multipliers_accepted(tmp_path):
    cfg = _load(tmp_path, {"scoring": {"match_multipliers": {"yes": 2.0, "maybe": 1.0, "no": 0.25}}})

    assert cfg.scoring.match_multiplier_yes == 2.0
    assert cfg.scoring.match_multiplier_no == 0.25


# ---------------------------------------------------------------------------
# Weather comfort configuration
# ---------------------------------------------------------------------------


_WEATHER_BLOCK = {
    "provider": "open-meteo",
    "default_hour": 19,
    "max_positive_adjustment": 0.15,
    "max_negative_adjustment": 0.25,
    "air_quality": {"enabled": False},
    "comfort": {
        "temperature_f": {
            "ideal": [20, 65],
            "zero": [-15, 78],
            "floor": [-40, 95],
            "weight": 1.0,
        },
        "dew_point_f": {
            "ideal": [-99, 55],
            "zero": [-99, 65],
            "floor": [-99, 75],
        },
        "relative_humidity": {
            "ideal": [0, 45],
            "zero": [0, 70],
            "floor": [0, 90],
            "fallback_for": "dew_point_f",
        },
        "precipitation_mm": {
            "ideal": [0, 0.3],
            "zero": [0, 2.5],
            "floor": [0, 10],
            "supersedes": ["rain", "snow"],
        },
    },
    "condition_penalty": {"rain": -0.4, "thunderstorm": -1.0},
}


def _weather_config(tmp_path, **overrides):
    block = {**_WEATHER_BLOCK, **overrides}
    return _load(tmp_path, {"weather": block}).weather


def test_weather_scalars_load(tmp_path):
    weather = _weather_config(tmp_path)
    assert weather.default_hour == 19
    assert weather.max_positive_adjustment == 0.15
    assert weather.max_negative_adjustment == 0.25
    assert weather.air_quality_enabled is False


def test_comfort_curve_bounds_load_as_tuples(tmp_path):
    curve = _weather_config(tmp_path).comfort["temperature_f"]
    assert curve.ideal == (20.0, 65.0)
    assert curve.zero == (-15.0, 78.0)
    assert curve.floor == (-40.0, 95.0)


def test_curve_optional_fields_default(tmp_path):
    curve = _weather_config(tmp_path).comfort["temperature_f"]
    assert curve.weight == 1.0
    assert curve.fallback_for is None
    assert curve.supersedes == ()


def test_curve_fallback_and_supersedes_load(tmp_path):
    comfort = _weather_config(tmp_path).comfort
    assert comfort["relative_humidity"].fallback_for == "dew_point_f"
    assert comfort["precipitation_mm"].supersedes == ("rain", "snow")


def test_condition_penalty_loads(tmp_path):
    weather = _weather_config(tmp_path)
    assert weather.condition_penalty["rain"] == -0.4
    assert weather.condition_penalty["thunderstorm"] == -1.0


def test_missing_weather_section_uses_defaults(tmp_path):
    weather = _load(tmp_path, {}).weather
    assert weather.default_hour == 20
    assert weather.air_quality_enabled is True


@pytest.mark.parametrize(
    "bad_curve",
    [
        {"ideal": [20, 65], "zero": [30, 78], "floor": [-40, 95]},  # zero_lo inside ideal
        {"ideal": [20, 65], "zero": [-15, 60], "floor": [-40, 95]},  # zero_hi inside ideal
        {"ideal": [20, 65], "zero": [-15, 78], "floor": [-10, 95]},  # floor_lo inside zero
        {"ideal": [20, 65], "zero": [-15, 78], "floor": [-40, 70]},  # floor_hi inside zero
        {"ideal": [65, 20], "zero": [-15, 78], "floor": [-40, 95]},  # inverted band
    ],
)
def test_inverted_curve_bounds_rejected(tmp_path, bad_curve):
    with pytest.raises(ConfigError, match="temperature_f"):
        _weather_config(tmp_path, comfort={"temperature_f": bad_curve})


def test_curve_missing_a_required_bound_rejected(tmp_path):
    with pytest.raises(ConfigError, match="temperature_f"):
        _weather_config(tmp_path, comfort={"temperature_f": {"ideal": [20, 65]}})


@pytest.mark.parametrize("field_name", ["max_positive_adjustment", "max_negative_adjustment"])
def test_negative_adjustment_cap_rejected(tmp_path, field_name):
    with pytest.raises(ConfigError, match=field_name):
        _weather_config(tmp_path, **{field_name: -0.1})


def test_fallback_for_unknown_factor_rejected(tmp_path):
    """A typo here silently disables a factor, so it must fail loudly."""
    with pytest.raises(ConfigError, match="dewpoint_f"):
        _weather_config(
            tmp_path,
            comfort={
                "relative_humidity": {
                    "ideal": [0, 45],
                    "zero": [0, 70],
                    "floor": [0, 90],
                    "fallback_for": "dewpoint_f",
                }
            },
        )


@pytest.mark.parametrize("bad_hour", [-1, 24])
def test_default_hour_outside_the_day_rejected(tmp_path, bad_hour):
    with pytest.raises(ConfigError, match="default_hour"):
        _weather_config(tmp_path, default_hour=bad_hour)


def test_cache_ttl_hours_loads(tmp_path):
    assert _weather_config(tmp_path, cache_ttl_hours=6).cache_ttl_hours == 6.0


def test_cache_ttl_hours_defaults_below_a_day(tmp_path):
    """A nightly batch must refetch, or it scores on yesterday's forecast."""
    assert 0 < _load(tmp_path, {}).weather.cache_ttl_hours < 24


@pytest.mark.parametrize("bad_ttl", [0, -1])
def test_non_positive_cache_ttl_rejected(tmp_path, bad_ttl):
    with pytest.raises(ConfigError, match="cache_ttl_hours"):
        _weather_config(tmp_path, cache_ttl_hours=bad_ttl)


def test_synthetic_rule_setting_loads(tmp_path):
    cfg = _load(tmp_path, {
        "synthetic_activities": [
            {"name": "Walk", "conditions": {}, "tags": ["walking"],
             "summary": "A walk", "setting": "outdoor"},
        ]
    })
    assert cfg.synthetic_activities[0].setting == "outdoor"


def test_synthetic_rule_without_setting_defaults_to_unknown(tmp_path):
    cfg = _load(tmp_path, {
        "synthetic_activities": [
            {"name": "Walk", "conditions": {}, "tags": ["walking"], "summary": "A walk"},
        ]
    })
    assert cfg.synthetic_activities[0].setting == "unknown"


def test_synthetic_rule_setting_outside_the_enum_rejected(tmp_path):
    with pytest.raises(ConfigError, match="Walk"):
        _load(tmp_path, {
            "synthetic_activities": [
                {"name": "Walk", "conditions": {}, "tags": ["walking"],
                 "summary": "A walk", "setting": "outside"},
            ]
        })


def _calendar(**overrides):
    entry = {
        "name": "northshorenightout",
        "url": "https://calendar.google.com/calendar/ical/abc/public/basic.ics",
    }
    entry.update(overrides)
    return entry


def test_sources_absent_yields_no_calendars(tmp_path):
    cfg = _load(tmp_path, {})
    assert cfg.sources.ics_calendars == []


def test_ics_calendar_loads(tmp_path):
    cfg = _load(tmp_path, {"sources": {"ics_calendars": [_calendar()]}})

    assert len(cfg.sources.ics_calendars) == 1
    calendar = cfg.sources.ics_calendars[0]
    assert calendar.name == "northshorenightout"
    assert calendar.url.endswith("/public/basic.ics")


def test_ics_calendar_source_type_defaults_to_its_name(tmp_path):
    cfg = _load(tmp_path, {"sources": {"ics_calendars": [_calendar()]}})

    assert cfg.sources.ics_calendars[0].source_type == "northshorenightout"


def test_ics_calendar_source_type_can_be_overridden(tmp_path):
    cfg = _load(tmp_path, {
        "sources": {"ics_calendars": [_calendar(source_type="community_calendar")]}
    })

    assert cfg.sources.ics_calendars[0].source_type == "community_calendar"


def test_ics_calendar_fetch_interval_defaults(tmp_path):
    cfg = _load(tmp_path, {"sources": {"ics_calendars": [_calendar()]}})

    assert cfg.sources.ics_calendars[0].min_fetch_interval_hours == 6.0


def test_ics_calendar_fetch_interval_reads_from_config(tmp_path):
    cfg = _load(tmp_path, {
        "sources": {"ics_calendars": [_calendar(min_fetch_interval_hours=12)]}
    })

    assert cfg.sources.ics_calendars[0].min_fetch_interval_hours == 12.0


def test_ics_calendar_missing_url_rejected(tmp_path):
    """A calendar with no URL is unusable, and failing at load beats failing at 2am."""
    with pytest.raises(ConfigError, match="url"):
        _load(tmp_path, {"sources": {"ics_calendars": [{"name": "broken"}]}})


def test_ics_calendar_missing_name_rejected(tmp_path):
    with pytest.raises(ConfigError, match="name"):
        _load(tmp_path, {
            "sources": {"ics_calendars": [{"url": "https://example.com/c.ics"}]}
        })


def test_negative_fetch_interval_rejected(tmp_path):
    """A negative interval would defeat the politeness guard it exists to enforce."""
    with pytest.raises(ConfigError, match="min_fetch_interval_hours"):
        _load(tmp_path, {
            "sources": {"ics_calendars": [_calendar(min_fetch_interval_hours=-1)]}
        })


def test_multiple_calendars_load_in_order(tmp_path):
    cfg = _load(tmp_path, {
        "sources": {
            "ics_calendars": [
                _calendar(name="first"),
                _calendar(name="second", url="https://example.com/second.ics"),
            ]
        }
    })

    assert [c.name for c in cfg.sources.ics_calendars] == ["first", "second"]


def test_html_calendars_absent_yields_empty(tmp_path):
    cfg = _load(tmp_path, {})
    assert cfg.sources.html_calendars == []


def test_html_calendar_loads_alongside_ics(tmp_path):
    """A site can legitimately appear as both a feed and a listing page."""
    cfg = _load(tmp_path, {
        "sources": {
            "ics_calendars": [_calendar()],
            "html_calendars": [
                {"name": "northshorenightout", "url": "https://example.com/"}
            ],
        }
    })

    assert len(cfg.sources.ics_calendars) == 1
    assert len(cfg.sources.html_calendars) == 1
    assert cfg.sources.html_calendars[0].source_type == "northshorenightout"


def test_html_calendar_missing_url_rejected(tmp_path):
    with pytest.raises(ConfigError, match="HTML calendar"):
        _load(tmp_path, {"sources": {"html_calendars": [{"name": "broken"}]}})


def test_models_block_loads(tmp_path):
    cfg = _load(
        tmp_path,
        {
            "models": {
                "llm_extraction": "custom-extract",
                "llm_disambiguation": "custom-disambig",
                "embeddings": "custom-embed",
            }
        },
    )
    assert cfg.models.llm_extraction == "custom-extract"
    assert cfg.models.llm_disambiguation == "custom-disambig"
    assert cfg.models.embeddings == "custom-embed"


def test_models_absent_yields_defaults(tmp_path):
    cfg = _load(tmp_path, {})
    assert cfg.models.llm_extraction == "gemma4:e4b"
    assert cfg.models.llm_disambiguation == "gemma4:e2b"
    assert cfg.models.embeddings == "nomic-embed-text"


def test_partial_models_block_keeps_other_defaults(tmp_path):
    cfg = _load(tmp_path, {"models": {"llm_extraction": "custom-extract"}})
    assert cfg.models.llm_extraction == "custom-extract"
    assert cfg.models.llm_disambiguation == "gemma4:e2b"
    assert cfg.models.embeddings == "nomic-embed-text"


@pytest.mark.parametrize(
    "key", ["llm_extraction", "llm_disambiguation", "embeddings"]
)
@pytest.mark.parametrize("blank", ["", "   "])
def test_blank_model_name_rejected(tmp_path, key, blank):
    """An empty name would reach Ollama as a request for no model at all."""
    with pytest.raises(ConfigError, match=key):
        _load(tmp_path, {"models": {key: blank}})


def test_provider_defaults_come_from_the_config_defaults(tmp_path):
    """One source of truth: a provider's fallback cannot drift from config."""
    from src.processing.extraction import OllamaExtractionProvider

    cfg = _load(tmp_path, {})
    provider = OllamaExtractionProvider(client=object())
    assert provider._model == cfg.models.llm_extraction


def test_horizon_days_loads(tmp_path):
    cfg = _load(tmp_path, {"scraping": {"horizon_days": 45}})
    assert cfg.scraping.horizon_days == 45


def test_horizon_days_defaults(tmp_path):
    """Defaults to the reach of the calendar feeds the lookahead exists for.

    Measured: northshorenightout publishes ~39 days out, so 30 truncated it.
    """
    assert _load(tmp_path, {}).scraping.horizon_days == DEFAULT_HORIZON_DAYS
    assert DEFAULT_HORIZON_DAYS == 45


@pytest.mark.parametrize("bad", [0, -1])
def test_non_positive_horizon_rejected(tmp_path, bad):
    with pytest.raises(ConfigError, match="horizon_days"):
        _load(tmp_path, {"scraping": {"horizon_days": bad}})


def test_timezone_finder_is_reused_across_calls():
    """Building a TimezoneFinder costs ~0.7s; load_config must not pay it every call."""
    assert _timezone_finder() is _timezone_finder()


def test_repeated_loads_build_one_timezone_finder(tmp_path, monkeypatch):
    """Pins the call site, not just the helper: N loads must construct one finder."""
    real = _timezone_finder()
    built = []

    def _counting_finder():
        built.append(1)
        return real

    _timezone_finder.cache_clear()
    monkeypatch.setattr("src.config.TimezoneFinder", _counting_finder)
    try:
        _load(tmp_path, {})
        _load(tmp_path, {})
    finally:
        _timezone_finder.cache_clear()

    assert len(built) == 1


def test_a_feed_claiming_a_reserved_source_type_is_rejected(tmp_path):
    """`synthetic` exempts an event from extraction and from confidence scaling.

    A feed quietly inheriting both would have its LLM tags never refreshed and
    its thin evidence never discounted.
    """
    with pytest.raises(ConfigError, match="synthetic"):
        _load(
            tmp_path,
            {
                "sources": {
                    "ics_calendars": [
                        {
                            "name": "sneaky",
                            "url": "https://x/f.ics",
                            "source_type": "synthetic",
                        }
                    ]
                }
            },
        )


def test_an_ordinary_source_type_is_accepted(tmp_path):
    cfg = _load(
        tmp_path,
        {
            "sources": {
                "ics_calendars": [
                    {"name": "nsno", "url": "https://x/f.ics", "source_type": "nsno_cal"}
                ]
            }
        },
    )

    assert cfg.sources.ics_calendars[0].source_type == "nsno_cal"


def test_day_starts_at_is_a_top_level_key(tmp_path):
    """One key, one concept.

    Ingestion and the CLI must agree on which day it is, or a re-run late in
    the evening discards the events the CLI is still showing. That makes the
    rollover a system-wide fact, not a rendering preference.
    """
    cfg = _load(tmp_path, {"day_starts_at": "05:30"})

    assert cfg.day_starts_at == time(5, 30)


def test_day_starts_at_defaults_to_four_am(tmp_path):
    """The default has to be a small hour, not midnight.

    A listing that rolls over at 00:00 answers "what should we do tonight?" with
    tomorrow, and empties the evening still in progress.
    """
    cfg = load_config(config_path=_write_config(tmp_path, _valid_location_data()))

    assert cfg.day_starts_at == time(4, 0)


def test_day_starts_at_midnight_is_accepted(tmp_path):
    """`00:00` is a real choice: it restores plain calendar-day semantics."""
    cfg = _load(tmp_path, {"day_starts_at": "00:00"})

    assert cfg.day_starts_at == time(0, 0)


@pytest.mark.parametrize("bad", ["25:00", "tea time", "4", "", "04:00:00:00", "0400"])
def test_malformed_day_starts_at_rejected(tmp_path, bad):
    """A bad value must not silently revert — it decides which day is shown."""
    with pytest.raises(ConfigError, match="day_starts_at"):
        _load(tmp_path, {"day_starts_at": bad})


def test_unquoted_sexagesimal_day_starts_at_rejected(tmp_path):
    """YAML 1.1 reads an unquoted `4:00` as the integer 240.

    Writing it without quotes is an easy mistake and `04:00` survives it, so the
    trap only springs on single-digit hours. Rejecting the int is what stops a
    listing quietly rolling over at some hour nobody chose.
    """
    config_file = tmp_path / "config.yaml"
    data = _valid_location_data()
    config_file.write_text(yaml.dump(data) + "day_starts_at: 4:00\n")

    with pytest.raises(ConfigError, match="quoted"):
        load_config(config_path=config_file)


def _feed(tmp_path, **overrides):
    """Load a config carrying one ICS feed entry, and return that feed."""
    entry = {"name": "capeann", "url": "https://x/f.ics"}
    entry.update(overrides)
    loaded = _load(tmp_path, {"sources": {"ics_calendars": [entry]}})
    return loaded.sources.ics_calendars[0]


class TestFeedVenueDefaults:
    """A single-venue feed has nowhere to declare its venue.

    The `[Venue, City]` summary convention is one aggregator's; a cinema's own
    calendar just names the film. Every event then arrives venue-less, which
    costs blocklist matching, dedup, and the CLI's `Title — Venue` line.
    """

    def test_venue_defaults_to_none(self, tmp_path):
        assert _feed(tmp_path).venue is None

    def test_city_defaults_to_none(self, tmp_path):
        assert _feed(tmp_path).city is None

    def test_a_declared_venue_loads(self, tmp_path):
        assert (
            _feed(tmp_path, venue="Cape Ann Community Cinema").venue
            == "Cape Ann Community Cinema"
        )

    def test_a_declared_city_loads(self, tmp_path):
        assert _feed(tmp_path, city="Gloucester").city == "Gloucester"

    def test_a_blank_venue_is_rejected(self, tmp_path):
        """A blank is a typo, not a choice — it would silently attribute nothing."""
        with pytest.raises(ConfigError, match="venue"):
            _feed(tmp_path, venue="   ")


# ---------------------------------------------------------------------------
# LLM request timeout
# ---------------------------------------------------------------------------


def test_the_llm_timeout_has_a_default_generous_enough_for_cpu_inference(tmp_path):
    """Measured on the target VM: one extraction exceeds any 60-second budget."""
    cfg = _load(tmp_path, {})

    assert cfg.models.request_timeout_seconds >= 600


def test_the_llm_timeout_can_be_configured(tmp_path):
    cfg = _load(tmp_path, {"models": {"request_timeout_seconds": 120}})

    assert cfg.models.request_timeout_seconds == 120


def test_a_non_positive_llm_timeout_is_rejected(tmp_path):
    with pytest.raises(ConfigError):
        _load(tmp_path, {"models": {"request_timeout_seconds": 0}})


def test_an_unreadable_llm_timeout_is_rejected(tmp_path):
    with pytest.raises(ConfigError):
        _load(tmp_path, {"models": {"request_timeout_seconds": "soon"}})


def test_the_context_window_default_leaves_room_for_a_model_that_reasons(tmp_path):
    """4096, ollama's default, is filled by reasoning before any content lands."""
    cfg = _load(tmp_path, {})

    assert cfg.models.num_ctx >= 32768


def test_the_sampling_default_is_tight_enough_for_structured_output(tmp_path):
    """gemma4:e4b ships temperature 1, which is indefensible for JSON."""
    cfg = _load(tmp_path, {})

    assert cfg.models.temperature <= 0.3
    assert 0 < cfg.models.top_p <= 1.0


def test_thinking_is_off_and_json_is_demanded_by_default(tmp_path):
    cfg = _load(tmp_path, {})

    assert cfg.models.think is False
    assert cfg.models.response_format == "json"


def test_models_are_released_rather_than_pinned_forever(tmp_path):
    """This host's server default never expires, so a model loaded once squats."""
    cfg = _load(tmp_path, {})

    assert cfg.models.keep_alive == "30m"


def test_keep_alive_can_be_configured(tmp_path):
    cfg = _load(tmp_path, {"models": {"keep_alive": "5m"}})

    assert cfg.models.keep_alive == "5m"


def test_a_blank_keep_alive_defers_to_the_server(tmp_path):
    cfg = _load(tmp_path, {"models": {"keep_alive": ""}})

    assert cfg.models.keep_alive is None


def test_the_request_parameters_can_be_configured(tmp_path):
    cfg = _load(
        tmp_path,
        {
            "models": {
                "temperature": 0.7,
                "top_p": 0.5,
                "num_ctx": 65536,
                "think": True,
                "format": "",
            }
        },
    )

    assert cfg.models.temperature == 0.7
    assert cfg.models.top_p == 0.5
    assert cfg.models.num_ctx == 65536
    assert cfg.models.think is True
    assert cfg.models.response_format is None


@pytest.mark.parametrize(
    "key, value",
    [
        ("num_ctx", 0),
        ("num_ctx", -1),
        ("num_ctx", "big"),
        ("temperature", -0.5),
        ("temperature", "hot"),
        ("top_p", 0),
        ("top_p", 1.5),
        ("top_p", "wide"),
    ],
)
def test_an_out_of_range_request_parameter_is_rejected(tmp_path, key, value):
    with pytest.raises(ConfigError):
        _load(tmp_path, {"models": {key: value}})


class TestSiteUrlBySourceType:
    """The human-facing page a source's events can fall back to.

    An event without its own URL is unattributable in the CLI, which is how a
    wrong listing looked like our mistake rather than the source's.
    """

    def _sources(self, feeds):
        return SourcesConfig(html_calendars=[FeedConfig(**f) for f in feeds])

    def test_a_feeds_url_stands_in_when_it_declares_no_site(self):
        sources = self._sources(
            [{"name": "pem", "url": "https://www.pem.org/events", "source_type": "pem"}]
        )

        assert sources.site_url_by_source_type() == {"pem": "https://www.pem.org/events"}

    def test_a_declared_site_wins_over_the_feed_url(self):
        # The NSNO case: the feed is a Google Calendar ICS link, useless to a
        # human who wants to check what the listing actually said.
        sources = self._sources(
            [{
                "name": "nsno",
                "url": "https://calendar.google.com/calendar/ical/abc/public/basic.ics",
                "source_type": "northshorenightout",
                "site_url": "https://northshorenightout.com/",
            }]
        )

        assert sources.site_url_by_source_type() == {
            "northshorenightout": "https://northshorenightout.com/"
        }

    def test_feeds_sharing_a_source_type_and_a_site_collapse_to_one_entry(self):
        sources = self._sources(
            [
                {"name": "a", "url": "https://do617.com/venues/koto",
                 "source_type": "do617", "site_url": "https://do617.com/"},
                {"name": "b", "url": "https://do617.com/venues/notch",
                 "source_type": "do617", "site_url": "https://do617.com/"},
            ]
        )

        assert sources.site_url_by_source_type() == {"do617": "https://do617.com/"}

    def test_feeds_disagreeing_on_the_site_yield_no_entry_at_all(self):
        # Two Veezi cinemas share `cinema_veezi`. Picking either would send half
        # the showings to the wrong cinema — worse than offering no link.
        sources = self._sources(
            [
                {"name": "warwick", "url": "https://veezi.example/?siteToken=aaa",
                 "source_type": "cinema_veezi"},
                {"name": "salem", "url": "https://veezi.example/?siteToken=bbb",
                 "source_type": "cinema_veezi"},
            ]
        )

        assert sources.site_url_by_source_type() == {}

    def test_every_declared_feed_list_is_searched_not_just_the_first(self):
        sources = SourcesConfig(
            ics_calendars=[FeedConfig(name="i", url="https://i.example/f.ics",
                                      source_type="ics_one")],
            veezi_cinemas=[FeedConfig(name="v", url="https://v.example/s",
                                      source_type="veezi_one")],
        )

        assert sources.site_url_by_source_type() == {
            "ics_one": "https://i.example/f.ics",
            "veezi_one": "https://v.example/s",
        }
