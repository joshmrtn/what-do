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
