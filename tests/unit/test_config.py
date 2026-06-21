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
