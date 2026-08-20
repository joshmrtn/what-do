"""A credential is registered for redaction at the moment it is read.

Credentials enter this process by **two doors** — `build_dependencies` reads
Apify's, AMC's and TMDb's from the injected environment, and `load_config` reads
Gemini's from `os.environ`. A guard written against one door is a guard that
misses the other, and that is not hypothetical: the first draft of this work
named only the adapters.

**What is being asserted, and why it is the door rather than the adapter.**
Minting a `Secret` *is* registering it, so the whole of the value layer follows
from where credentials are minted. An adapter handed a `Secret` by a test is
protected by the test's own choice of type; a credential that came through the
door is protected by `src/`. Only the second says anything.

The assertion is made on text of the shape a credential really escapes in —
`str(requests.HTTPError)` embeds the fully-built URL, query string included, so
the value arrives inside a message nothing we wrote composed.

Two controls keep it honest. A value no door ever saw must survive `scrub`
untouched, or a registry that redacted everything would pass. And `OLLAMA_HOST`
goes through the same door as `GEMINI_API_KEY` and must come back intact — it is
a URL with no auth in it, so registering it would be the design failing in the
other direction.
"""

from __future__ import annotations

import io
import json

import pytest
import yaml

from src.composition.batch import build_dependencies
from src.config import load_config
from src.storage.sqlite.connection import init_db
from src.utils.logging import get_logger
from src.utils.secret import scrub
from tests.unit.test_composition import _config

#: Well clear of `MIN_REDACTABLE_LENGTH`, and unmistakable in a haystack.
SENTINELS = {
    "APIFY_API_KEY": "apify-sentinel-4d9f2a7c1e",
    "AMC_API_KEY": "amc-sentinel-b83c1f6e02",
    "TMDB_READ_ACCESS_TOKEN": "tmdb-sentinel-7a41d0c95f",
}

GEMINI_SENTINEL = "gemini-sentinel-2c5e8b1943"

#: Never handed to any door. Proves the assertion below can fail.
UNREGISTERED = "never-offered-06f4b7e2ad"


def _provider_error(value: str) -> str:
    """Text of the shape a credential actually escapes in."""
    return (
        "401 Client Error: Unauthorized for url: "
        f"https://api.example.test/v2/thing?token={value}&usernames=somebar"
    )


@pytest.fixture
def paths(tmp_path) -> dict:
    db = tmp_path / "batch.db"
    init_db(db)
    (tmp_path / "seeds.yaml").write_text("handles: ['@jazzclub']\nvenues: []\n")
    (tmp_path / "likes.txt").write_text("")
    (tmp_path / "dislikes.txt").write_text("")
    (tmp_path / "blocklist.json").write_text(json.dumps([]))
    return {
        "db_path": db,
        "seeds_path": tmp_path / "seeds.yaml",
        "likes_path": tmp_path / "likes.txt",
        "dislikes_path": tmp_path / "dislikes.txt",
        "blocklist_path": tmp_path / "blocklist.json",
    }


def _config_file(tmp_path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        yaml.dump(
            {
                "location": {
                    "latitude": 42.52,
                    "longitude": -70.89,
                    "postal_code": "01970",
                    "search_radius_miles": 10,
                }
            }
        )
    )
    return config_file


@pytest.mark.parametrize("variable", sorted(SENTINELS))
def test_the_batch_door_registers_every_credential_it_reads(paths, variable):
    build_dependencies(
        config=_config(),
        logger=get_logger("credential_door_test", stream=io.StringIO()),
        env=dict(SENTINELS),
        **paths,
    )

    sentinel = SENTINELS[variable]
    assert sentinel not in scrub(_provider_error(sentinel)), (
        f"{variable} reached the composition root without becoming a Secret, "
        "so nothing will scrub it out of a failure the provider reports"
    )


def test_the_config_door_registers_the_gemini_credential(tmp_path, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", GEMINI_SENTINEL)

    load_config(config_path=_config_file(tmp_path), env_path=tmp_path / "absent.env")

    assert GEMINI_SENTINEL not in scrub(_provider_error(GEMINI_SENTINEL))


def test_a_value_no_door_ever_read_is_left_alone(paths):
    """Without this, a registry that redacted everything would pass above."""
    build_dependencies(
        config=_config(),
        logger=get_logger("credential_door_test", stream=io.StringIO()),
        env=dict(SENTINELS),
        **paths,
    )

    assert UNREGISTERED in scrub(_provider_error(UNREGISTERED))


def test_the_ollama_host_is_not_treated_as_a_credential(tmp_path, monkeypatch):
    """It comes through the same door and is a URL with no auth in it.

    Registering it would redact the host out of every log line that names it,
    which is this design failing in the direction that is merely noisy rather
    than dangerous — but still failing.
    """
    host = "http://gpu-box-9f3a2c7e5b:11434"
    monkeypatch.setenv("OLLAMA_HOST", host)

    load_config(config_path=_config_file(tmp_path), env_path=tmp_path / "absent.env")

    assert host in scrub(f"embedding call to {host} timed out")
