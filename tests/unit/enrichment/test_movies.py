"""Unit tests for MovieMetadataProvider and enrich_movie_event."""

from datetime import datetime, timezone
from unittest.mock import MagicMock, call

import pytest

from src.enrichment.movies import (
    MovieMetadataProvider,
    TMDbProvider,
    enrich_movie_event,
)
from src.models.event import Event
from src.utils.logging import StructuredLogger

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

NOW = datetime(2025, 6, 21, 12, 0, tzinfo=timezone.utc)

_METADATA = {
    "genres": ["Horror", "Thriller"],
    "runtime_minutes": 112,
    "summary": "A terrifying tale.",
    "release_year": 2024,
}


def _movie_event(source_type: str = "cinema_veezi", title: str | None = "The Thing") -> Event:
    return Event(
        event_id="evt-1",
        source_event_candidates=[],
        source_type=source_type,
        created_at=NOW,
        updated_at=NOW,
        title=title,
    )


def _mock_logger() -> StructuredLogger:
    logger = MagicMock(spec=StructuredLogger)
    return logger


def _mock_provider(return_value=_METADATA, side_effect=None) -> MovieMetadataProvider:
    provider = MagicMock(spec=MovieMetadataProvider)
    if side_effect is not None:
        provider.fetch.side_effect = side_effect
    else:
        provider.fetch.return_value = return_value
    return provider


# ---------------------------------------------------------------------------
# ABC conformance
# ---------------------------------------------------------------------------


def test_movie_metadata_provider_is_abstract():
    with pytest.raises(TypeError):
        MovieMetadataProvider()


def test_tmdb_provider_is_movie_metadata_provider():
    provider = TMDbProvider(api_key="key", session=MagicMock())
    assert isinstance(provider, MovieMetadataProvider)


# ---------------------------------------------------------------------------
# enrich_movie_event — happy path
# ---------------------------------------------------------------------------


def test_metadata_merged_into_event_metadata():
    event = _movie_event()
    provider = _mock_provider()
    result = enrich_movie_event(event, provider, _mock_logger())
    assert result.metadata["genres"] == ["Horror", "Thriller"]
    assert result.metadata["runtime_minutes"] == 112
    assert result.metadata["summary"] == "A terrifying tale."
    assert result.metadata["release_year"] == 2024


def test_returns_same_event_object():
    event = _movie_event()
    result = enrich_movie_event(event, _mock_provider(), _mock_logger())
    assert result is event


def test_provider_called_with_event_title():
    event = _movie_event(title="The Thing")
    provider = _mock_provider()
    enrich_movie_event(event, provider, _mock_logger())
    provider.fetch.assert_called_once()
    args, _ = provider.fetch.call_args
    assert args[0] == "The Thing"


# ---------------------------------------------------------------------------
# enrich_movie_event — provider returns None
# ---------------------------------------------------------------------------


def test_provider_returns_none_metadata_unchanged():
    event = _movie_event()
    event.metadata["existing_key"] = "value"
    provider = _mock_provider(return_value=None)
    result = enrich_movie_event(event, provider, _mock_logger())
    assert result.metadata == {"existing_key": "value"}


def test_provider_returns_none_warning_logged():
    logger = _mock_logger()
    enrich_movie_event(_movie_event(), _mock_provider(return_value=None), logger)
    logger.warning.assert_called_once()


# ---------------------------------------------------------------------------
# enrich_movie_event — provider raises exception
# ---------------------------------------------------------------------------


def test_provider_raises_metadata_unchanged():
    event = _movie_event()
    event.metadata["existing_key"] = "value"
    provider = _mock_provider(side_effect=RuntimeError("TMDb down"))
    result = enrich_movie_event(event, provider, _mock_logger())
    assert result.metadata == {"existing_key": "value"}


def test_provider_raises_error_logged():
    logger = _mock_logger()
    provider = _mock_provider(side_effect=RuntimeError("TMDb down"))
    enrich_movie_event(_movie_event(), provider, logger)
    logger.error.assert_called_once()


def test_provider_raises_pipeline_continues():
    """Exception does not propagate — caller should not see it."""
    provider = _mock_provider(side_effect=RuntimeError("TMDb down"))
    # Should not raise
    enrich_movie_event(_movie_event(), provider, _mock_logger())


# ---------------------------------------------------------------------------
# enrich_movie_event — guard: non-movie source_type
# ---------------------------------------------------------------------------


def test_non_movie_event_provider_not_called():
    event = _movie_event(source_type="instagram")
    provider = _mock_provider()
    enrich_movie_event(event, provider, _mock_logger())
    provider.fetch.assert_not_called()


def test_non_movie_event_metadata_unchanged():
    event = _movie_event(source_type="instagram")
    enrich_movie_event(event, _mock_provider(), _mock_logger())
    assert event.metadata == {}


def test_amc_event_provider_is_called():
    event = _movie_event(source_type="amc")
    provider = _mock_provider()
    enrich_movie_event(event, provider, _mock_logger())
    provider.fetch.assert_called_once()


# ---------------------------------------------------------------------------
# enrich_movie_event — guard: missing title
# ---------------------------------------------------------------------------


def test_movie_event_with_no_title_provider_not_called():
    event = _movie_event(title=None)
    provider = _mock_provider()
    enrich_movie_event(event, provider, _mock_logger())
    provider.fetch.assert_not_called()


def test_movie_event_with_no_title_metadata_unchanged():
    event = _movie_event(title=None)
    enrich_movie_event(event, _mock_provider(), _mock_logger())
    assert event.metadata == {}


# ---------------------------------------------------------------------------
# TMDbProvider — with mocked session
# ---------------------------------------------------------------------------


def _tmdb_session(search_results=None, detail_result=None):
    """Build a mock session that returns canned TMDb API responses."""
    search_resp = MagicMock()
    search_resp.raise_for_status.return_value = None
    search_resp.json.return_value = {
        "results": search_results
        if search_results is not None
        else [{"id": 99, "release_date": "2024-10-01"}]
    }

    detail_resp = MagicMock()
    detail_resp.raise_for_status.return_value = None
    detail_resp.json.return_value = detail_result or {
        "genres": [{"name": "Horror"}, {"name": "Thriller"}],
        "runtime": 112,
        "overview": "A terrifying tale.",
        "release_date": "2024-10-01",
    }

    session = MagicMock()
    session.get.side_effect = [search_resp, detail_resp]
    return session


def test_tmdb_provider_returns_metadata_dict():
    provider = TMDbProvider(api_key="testkey", session=_tmdb_session())
    result = provider.fetch("The Thing", year=None)
    assert result is not None
    assert set(result.keys()) == {"genres", "runtime_minutes", "summary", "release_year"}


def test_tmdb_provider_genres_are_list_of_strings():
    provider = TMDbProvider(api_key="testkey", session=_tmdb_session())
    result = provider.fetch("The Thing", year=None)
    assert isinstance(result["genres"], list)
    assert all(isinstance(g, str) for g in result["genres"])


def test_tmdb_provider_runtime_minutes():
    provider = TMDbProvider(api_key="testkey", session=_tmdb_session())
    result = provider.fetch("The Thing", year=None)
    assert result["runtime_minutes"] == 112


def test_tmdb_provider_summary():
    provider = TMDbProvider(api_key="testkey", session=_tmdb_session())
    result = provider.fetch("The Thing", year=None)
    assert result["summary"] == "A terrifying tale."


def test_tmdb_provider_release_year():
    provider = TMDbProvider(api_key="testkey", session=_tmdb_session())
    result = provider.fetch("The Thing", year=None)
    assert result["release_year"] == 2024


def test_tmdb_provider_no_results_returns_none():
    session = _tmdb_session(search_results=[])
    # Only one call (search), no detail call needed
    single_resp = MagicMock()
    single_resp.raise_for_status.return_value = None
    single_resp.json.return_value = {"results": []}
    session = MagicMock()
    session.get.return_value = single_resp
    provider = TMDbProvider(api_key="testkey", session=session)
    assert provider.fetch("Nonexistent Movie", year=None) is None


def test_tmdb_provider_network_error_returns_none():
    session = MagicMock()
    session.get.side_effect = Exception("network error")
    provider = TMDbProvider(api_key="testkey", session=session)
    assert provider.fetch("The Thing", year=None) is None


def test_tmdb_provider_missing_runtime_returns_none_for_that_field():
    session = _tmdb_session(
        detail_result={
            "genres": [{"name": "Drama"}],
            "runtime": None,
            "overview": "A drama.",
            "release_date": "2023-05-15",
        }
    )
    provider = TMDbProvider(api_key="testkey", session=session)
    result = provider.fetch("A Drama", year=None)
    assert result["runtime_minutes"] is None
