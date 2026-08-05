import os

import pytest
import yaml

from src.config import ConfigError, load_config


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
    assert cfg.scoring.top_picks_min == 0.5
    assert cfg.scoring.worth_considering_min == 0.1
    assert cfg.scoring.summary_weight == 0.3
    assert cfg.scoring.match_multiplier_yes == 1.5
    assert cfg.scoring.match_multiplier_maybe == 1.0
    assert cfg.scoring.match_multiplier_no == 0.5
    assert cfg.scoring.min_tags_per_event == 5


def test_scoring_reads_from_config(tmp_path):
    data = _valid_location_data()
    data["scoring"] = {
        "tiers": {"top_picks_min": 0.7, "worth_considering_min": 0.2},
        "summary_weight": 0.4,
        "match_multipliers": {"yes": 2.0, "maybe": 1.0, "no": 0.25},
        "min_tags_per_event": 8,
    }
    cfg = load_config(config_path=_write_config(tmp_path, data))
    assert cfg.scoring.top_picks_min == 0.7
    assert cfg.scoring.worth_considering_min == 0.2
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
