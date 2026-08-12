from __future__ import annotations

import io
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pytest

from src.composition import build_dependencies
from src.config import (
    AppConfig,
    DeduplicationConfig,
    FeedConfig,
    LocationConfig,
    ModelsConfig,
    ScrapingConfig,
    SourcesConfig,
    VenueDiscoveryConfig,
)
from src.ingestion.aggregators.do617_source import Do617VenueSource
from src.ingestion.aggregators.jsonld_source import JsonLdEventSource
from src.ingestion.calendars.assabet_source import AssabetRssSource
from src.ingestion.calendars.html_source import HtmlListingSource
from src.ingestion.calendars.ics_source import IcsCalendarSource
from src.ingestion.calendars.moon_source import MoonRssSource
from src.models.event import Event
from src.models.event_candidate import EventCandidate
from src.storage.sqlite.connection import init_db
from src.utils.logging import get_logger

FULL_ENV = {
    "APIFY_API_KEY": "apify-key",
    "TMDB_API_KEY": "tmdb-key",
    "AMC_API_KEY": "amc-key",
}


def _config(**overrides) -> AppConfig:
    fields = {
        "location": LocationConfig(42.52, -70.89, "01970", 10.0, "America/New_York"),
        "scraping": ScrapingConfig(),
        "venue_discovery": VenueDiscoveryConfig(),
        "deduplication": DeduplicationConfig(),
    }
    fields.update(overrides)
    return AppConfig(**fields)


@pytest.fixture
def paths(tmp_path) -> dict:
    db = tmp_path / "batch.db"
    init_db(db)
    (tmp_path / "seeds.yaml").write_text("handles: ['@jazzclub']\nvenues: []\n")
    (tmp_path / "likes.txt").write_text("")
    (tmp_path / "dislikes.txt").write_text("")
    (tmp_path / "blocklist.json").write_text(json.dumps(["Some Bar", "@spammer"]))
    return {
        "db_path": db,
        "seeds_path": tmp_path / "seeds.yaml",
        "likes_path": tmp_path / "likes.txt",
        "dislikes_path": tmp_path / "dislikes.txt",
        "blocklist_path": tmp_path / "blocklist.json",
    }


def _build(paths, env=None, config=None, stream=None):
    return build_dependencies(
        config=config or _config(),
        logger=get_logger("composition_test", stream=stream or io.StringIO()),
        env=FULL_ENV if env is None else env,
        **paths,
    )


# ----------------------------------------------------------------------
# Credentials
# ----------------------------------------------------------------------


def test_every_credential_present_skips_nothing(paths):
    assert _build(paths).skipped_sources == []


def test_a_source_with_no_credential_is_skipped(paths):
    env = {k: v for k, v in FULL_ENV.items() if k != "APIFY_API_KEY"}

    assert "apify" in _build(paths, env=env).skipped_sources


def test_the_warning_names_the_variable_it_looked_for(paths):
    """A mistyped variable name is the one genuinely silent case; name it."""
    stream = io.StringIO()
    env = {k: v for k, v in FULL_ENV.items() if k != "APIFY_API_KEY"}

    _build(paths, env=env, stream=stream)

    assert "APIFY_API_KEY" in stream.getvalue()


def test_a_blank_credential_counts_as_absent(paths):
    """.env.example ships every key blank, so copying it leaves real blanks."""
    env = dict(FULL_ENV, APIFY_API_KEY="")

    assert "apify" in _build(paths, env=env).skipped_sources


def test_no_credentials_at_all_still_builds(paths):
    """A user with no keys still gets the calendar sources and the full pipeline."""
    deps = _build(paths, env={})

    assert deps.ingestion_service is not None
    assert sorted(deps.skipped_sources) == ["amc", "apify", "tmdb"]


def test_a_missing_tmdb_key_leaves_enrichment_without_a_movie_provider(paths):
    env = {k: v for k, v in FULL_ENV.items() if k != "TMDB_API_KEY"}

    deps = _build(paths, env=env)

    assert deps.enrichment_service._movie_provider is None


# ----------------------------------------------------------------------
# Config-declared sources
# ----------------------------------------------------------------------


def test_calendar_sources_are_built_from_config(paths):
    config = _config(
        sources=SourcesConfig(
            ics_calendars=[FeedConfig("nsno", "https://x/feed.ics", "nsno_cal")],
            html_calendars=[FeedConfig("nsno_list", "https://x/events", "nsno_list")],
        )
    )

    sources = _build(paths, config=config).ingestion_service._independent_sources

    assert any(isinstance(s, IcsCalendarSource) for s in sources)
    assert any(isinstance(s, HtmlListingSource) for s in sources)


def test_do617_venue_sources_are_built_from_config(paths):
    config = _config(
        sources=SourcesConfig(
            do617_venues=[
                FeedConfig("do617_gulu_gulu", "https://do617.com/venues/gulu-gulu-cafe", "do617")
            ]
        )
    )

    sources = _build(paths, config=config).ingestion_service._independent_sources

    assert any(isinstance(s, Do617VenueSource) for s in sources)


def test_jsonld_sources_are_built_from_config(paths):
    config = _config(
        sources=SourcesConfig(
            jsonld_pages=[FeedConfig("pem", "https://www.pem.org/events", "pem")]
        )
    )

    sources = _build(paths, config=config).ingestion_service._independent_sources

    assert any(isinstance(s, JsonLdEventSource) for s in sources)


def test_assabet_feed_sources_are_built_from_config(paths):
    config = _config(
        sources=SourcesConfig(
            assabet_feeds=[FeedConfig("salempl", "https://x.assabetinteractive.com/f.rss", "salempl")]
        )
    )

    sources = _build(paths, config=config).ingestion_service._independent_sources

    assert any(isinstance(s, AssabetRssSource) for s in sources)


def test_moon_feed_sources_are_built_from_config(paths):
    config = _config(
        sources=SourcesConfig(
            moon_feeds=[FeedConfig("moon", "https://www.moon-ns.org/shows?format=rss", "moon")]
        )
    )

    sources = _build(paths, config=config).ingestion_service._independent_sources

    assert any(isinstance(s, MoonRssSource) for s in sources)


def test_calendar_sources_need_no_credential(paths):
    config = _config(
        sources=SourcesConfig(
            ics_calendars=[FeedConfig("nsno", "https://x/feed.ics", "nsno_cal")]
        )
    )

    deps = _build(paths, env={}, config=config)

    assert any(
        isinstance(s, IcsCalendarSource)
        for s in deps.ingestion_service._independent_sources
    )


# ----------------------------------------------------------------------
# Model names come from config, never hardcoded
# ----------------------------------------------------------------------


def test_model_names_come_from_config(paths):
    config = _config(
        models=ModelsConfig(
            llm_extraction="custom-extract",
            llm_disambiguation="custom-disambig",
            embeddings="custom-embed",
        )
    )

    deps = _build(paths, config=config)

    assert deps.extraction_stage._provider._model == "custom-extract"
    assert deps.embedding_stage._provider._model == "custom-embed"


# ----------------------------------------------------------------------
# Blocklist
# ----------------------------------------------------------------------


def test_the_blocklist_reaches_both_readers(paths):
    """Handles are enforced at ingestion, venue names at ranking. See #15."""
    deps = _build(paths)

    assert "Some Bar" in deps.ingestion_service._blocklist
    assert "Some Bar" in deps.ranking_engine._blocklist


def test_an_absent_blocklist_file_is_not_an_error(paths, tmp_path):
    """A user who never wrote one is a normal deployment, not a failure."""
    deps = _build(dict(paths, blocklist_path=tmp_path / "nope.json"))

    assert deps.ranking_engine._blocklist == []


def test_the_llm_timeout_reaches_the_ollama_client(paths):
    """A 60-second default failed every extraction on the batch VM."""
    config = _config(models=ModelsConfig(request_timeout_seconds=900))

    deps = _build(paths, config=config)

    assert deps.extraction_stage._provider._client._timeout == 900


class TestPersistenceIsWiredToTheDatabase:
    """Composition is now the only place a concrete repository is named.

    `run_batch` used to fall back to constructing its own SQLite repository
    whenever one was not injected, so composition wiring the wrong thing was
    harmless. Removing those fallbacks made this the single point of failure,
    and nothing here was checking it: swapping the event repository for an
    in-memory one — a batch that persists nothing, all night, every night —
    left all 2056 tests green.

    Asserted through behaviour rather than by type, so a repository that is
    the right class but points at the wrong database still fails.
    """

    def _event(self) -> Event:
        now = datetime(2026, 6, 15, 12, 0, tzinfo=timezone.utc)
        return Event(
            event_id="wired-1",
            source_event_candidates=[],
            source_type="apify",
            created_at=now,
            updated_at=now,
            title="Karaoke Night",
        )

    def test_events_reach_the_database_on_disk(self, paths):
        deps = _build(paths)

        deps.event_repository.save([self._event()])

        conn = sqlite3.connect(paths["db_path"])
        try:
            titles = [r[0] for r in conn.execute("SELECT title FROM events")]
        finally:
            conn.close()
        assert titles == ["Karaoke Night"]

    def test_runs_reach_the_database_on_disk(self, paths):
        deps = _build(paths)

        run_id = deps.run_repository.start(
            datetime(2026, 6, 15, 12, 0, tzinfo=timezone.utc)
        )

        conn = sqlite3.connect(paths["db_path"])
        try:
            rows = [r[0] for r in conn.execute("SELECT id FROM run_history")]
        finally:
            conn.close()
        assert rows == [run_id]

    def test_candidates_reach_the_database_on_disk(self, paths):
        deps = _build(paths)

        deps.candidate_repository.save(
            [
                EventCandidate(
                    id="cand-1",
                    source="@venue",
                    source_type="apify",
                    discovered_at=datetime(2026, 6, 15, 12, 0, tzinfo=timezone.utc),
                    title="Quiz Night",
                )
            ]
        )

        conn = sqlite3.connect(paths["db_path"])
        try:
            rows = [r[0] for r in conn.execute("SELECT id FROM event_candidates")]
        finally:
            conn.close()
        assert rows == ["cand-1"]
