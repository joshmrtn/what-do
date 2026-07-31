"""Unit tests for ExtractionStage."""

from __future__ import annotations

import io
from datetime import datetime, timezone
from unittest.mock import MagicMock, call

import pytest

from src.models.event import Event
from src.utils.logging import get_logger


def _now() -> datetime:
    return datetime(2026, 6, 22, 12, 0, 0, tzinfo=timezone.utc)


def _make_event(**kwargs) -> Event:
    defaults = dict(
        event_id="evt-1",
        source_event_candidates=["cand-1"],
        source_type="apify",
        created_at=_now(),
        updated_at=_now(),
        title="Live Jazz Night",
        description="Come enjoy live jazz at the waterfront venue this Saturday.",
    )
    defaults.update(kwargs)
    return Event(**defaults)


def _make_logger():
    return get_logger("test_stage", stream=io.StringIO())


def _make_provider(tags=None, summary="A great event."):
    from src.processing.extraction import ExtractionResult

    provider = MagicMock()
    provider.extract.return_value = ExtractionResult(
        title="Extracted Title",
        venue="Extracted Venue",
        start_time=datetime(2026, 6, 22, 20, 0, 0),
        end_time=None,
        tags=tags or ["jazz", "live music", "evening", "venue", "weekend"],
        summary=summary,
    )
    return provider


def _make_fetcher(content: bytes = b"fake image bytes"):
    fetcher = MagicMock()
    fetcher.fetch.return_value = content
    return fetcher


# ---------------------------------------------------------------------------
# Bypass
# ---------------------------------------------------------------------------


def test_bypass_when_tags_already_populated():
    from src.processing.extraction_stage import ExtractionStage

    event = _make_event(tags=["jazz", "live", "music", "fun", "night"])
    provider = _make_provider()
    stage = ExtractionStage(provider=provider, image_fetcher=None, logger=_make_logger())

    results = stage.process([event])

    provider.extract.assert_not_called()
    assert results[0].tags == ["jazz", "live", "music", "fun", "night"]


def test_extraction_called_when_tags_empty():
    from src.processing.extraction_stage import ExtractionStage

    event = _make_event(tags=[])
    provider = _make_provider()
    stage = ExtractionStage(provider=provider, image_fetcher=None, logger=_make_logger())

    stage.process([event])

    provider.extract.assert_called_once()


# ---------------------------------------------------------------------------
# Field merging
# ---------------------------------------------------------------------------


def test_tags_and_summary_always_written():
    from src.processing.extraction_stage import ExtractionStage

    event = _make_event()
    provider = _make_provider(tags=["a", "b", "c", "d", "e"], summary="Excellent event.")
    stage = ExtractionStage(provider=provider, image_fetcher=None, logger=_make_logger())

    results = stage.process([event])

    assert results[0].tags == ["a", "b", "c", "d", "e"]
    assert results[0].summary == "Excellent event."


def test_llm_fills_null_title():
    from src.processing.extraction_stage import ExtractionStage

    event = _make_event(title=None)
    provider = _make_provider()
    stage = ExtractionStage(provider=provider, image_fetcher=None, logger=_make_logger())

    results = stage.process([event])

    assert results[0].title == "Extracted Title"


def test_llm_does_not_overwrite_existing_title():
    from src.processing.extraction_stage import ExtractionStage

    event = _make_event(title="Existing Title From Normalization")
    provider = _make_provider()
    stage = ExtractionStage(provider=provider, image_fetcher=None, logger=_make_logger())

    results = stage.process([event])

    assert results[0].title == "Existing Title From Normalization"


def test_llm_fills_null_venue():
    from src.processing.extraction_stage import ExtractionStage

    event = _make_event(venue=None)
    provider = _make_provider()
    stage = ExtractionStage(provider=provider, image_fetcher=None, logger=_make_logger())

    results = stage.process([event])

    assert results[0].venue == "Extracted Venue"


def test_llm_does_not_overwrite_existing_venue():
    from src.processing.extraction_stage import ExtractionStage

    event = _make_event(venue="The Vault Lounge")
    provider = _make_provider()
    stage = ExtractionStage(provider=provider, image_fetcher=None, logger=_make_logger())

    results = stage.process([event])

    assert results[0].venue == "The Vault Lounge"


def test_llm_fills_null_start_time():
    from src.processing.extraction_stage import ExtractionStage

    event = _make_event(start_time=None)
    provider = _make_provider()
    stage = ExtractionStage(provider=provider, image_fetcher=None, logger=_make_logger())

    results = stage.process([event])

    assert results[0].start_time == datetime(2026, 6, 22, 20, 0, 0)


def test_llm_does_not_overwrite_existing_start_time():
    from src.processing.extraction_stage import ExtractionStage

    existing = datetime(2026, 6, 22, 19, 30, 0, tzinfo=timezone.utc)
    event = _make_event(start_time=existing)
    provider = _make_provider()
    stage = ExtractionStage(provider=provider, image_fetcher=None, logger=_make_logger())

    results = stage.process([event])

    assert results[0].start_time == existing


# ---------------------------------------------------------------------------
# Image fetching
# ---------------------------------------------------------------------------


def test_image_url_triggers_fetch_and_sets_image_bytes():
    from src.processing.extraction_stage import ExtractionStage

    event = _make_event(image_url="http://example.com/photo.jpg")
    fetcher = _make_fetcher(b"real image bytes")
    provider = _make_provider()
    stage = ExtractionStage(provider=provider, image_fetcher=fetcher, logger=_make_logger())

    results = stage.process([event])

    fetcher.fetch.assert_called_once_with("http://example.com/photo.jpg")
    assert results[0].image_bytes == b"real image bytes"
    # provider should have received the image bytes
    provider.extract.assert_called_once()
    _, kwargs = provider.extract.call_args
    assert kwargs.get("image_bytes") == b"real image bytes"


def test_no_image_url_no_fetch():
    from src.processing.extraction_stage import ExtractionStage

    event = _make_event(image_url=None)
    fetcher = _make_fetcher()
    provider = _make_provider()
    stage = ExtractionStage(provider=provider, image_fetcher=fetcher, logger=_make_logger())

    stage.process([event])

    fetcher.fetch.assert_not_called()
    _, kwargs = provider.extract.call_args
    assert kwargs.get("image_bytes") is None


def test_image_fetch_error_logs_warning_and_continues():
    from src.processing.extraction_stage import ExtractionStage
    from src.processing.image_fetcher import ImageFetchError

    event = _make_event(image_url="http://example.com/broken.jpg")
    fetcher = MagicMock()
    fetcher.fetch.side_effect = ImageFetchError("404")
    provider = _make_provider()
    stage = ExtractionStage(provider=provider, image_fetcher=fetcher, logger=_make_logger())

    results = stage.process([event])

    # extraction still proceeds, but with no image bytes
    provider.extract.assert_called_once()
    _, kwargs = provider.extract.call_args
    assert kwargs.get("image_bytes") is None
    # event not dropped
    assert len(results) == 1


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


def test_extraction_error_sets_flag_and_continues():
    from src.processing.extraction import ExtractionError
    from src.processing.extraction_stage import ExtractionStage

    provider = MagicMock()
    provider.extract.side_effect = ExtractionError("LLM confused")
    stage = ExtractionStage(provider=provider, image_fetcher=None, logger=_make_logger())

    event = _make_event()
    results = stage.process([event])

    assert results[0].metadata.get("llm_extraction_failed") is True
    assert results[0].tags == []


def test_one_failed_event_does_not_stop_others():
    from src.processing.extraction import ExtractionError
    from src.processing.extraction_stage import ExtractionStage

    provider = MagicMock()
    provider.extract.side_effect = [
        ExtractionError("bad"),
        provider.extract.return_value,  # won't reach, side_effect overrides
    ]

    good_result_mock = MagicMock()
    good_result_mock.tags = ["a", "b", "c", "d", "e"]
    good_result_mock.summary = "Good event."
    good_result_mock.title = None
    good_result_mock.venue = None
    good_result_mock.start_time = None
    good_result_mock.end_time = None

    from src.processing.extraction import ExtractionError, ExtractionResult

    provider2 = MagicMock()
    provider2.extract.side_effect = [
        ExtractionError("bad"),
        ExtractionResult(
            title=None, venue=None, start_time=None, end_time=None,
            tags=["a", "b", "c", "d", "e"], summary="Good."
        ),
    ]

    event1 = _make_event(event_id="evt-fail")
    event2 = _make_event(event_id="evt-ok")

    stage = ExtractionStage(provider=provider2, image_fetcher=None, logger=_make_logger())
    results = stage.process([event1, event2])

    assert len(results) == 2
    assert results[0].metadata.get("llm_extraction_failed") is True
    assert results[1].tags == ["a", "b", "c", "d", "e"]


# ---------------------------------------------------------------------------
# Input text construction
# ---------------------------------------------------------------------------


def test_input_text_combines_title_and_description():
    from src.processing.extraction_stage import ExtractionStage

    event = _make_event(title="Jazz Night", description="Live jazz at the waterfront.")
    provider = _make_provider()
    stage = ExtractionStage(provider=provider, image_fetcher=None, logger=_make_logger())

    stage.process([event])

    text_arg = provider.extract.call_args[0][0]
    assert "Jazz Night" in text_arg
    assert "Live jazz at the waterfront." in text_arg


# ---------------------------------------------------------------------------
# Slow smoke test
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_real_extraction_produces_valid_result():
    """Confirm real Ollama extraction works end-to-end with gemma4:e4b."""
    from src.utils.ollama_client import OllamaClient
    from src.processing.extraction import OllamaExtractionProvider
    from src.processing.extraction_stage import ExtractionStage

    client = OllamaClient(host="http://localhost:11434", timeout=3600)
    provider = OllamaExtractionProvider(client=client, model="gemma4:e4b", min_tags=5)
    stage = ExtractionStage(provider=provider, image_fetcher=None, logger=_make_logger())

    event = _make_event(
        title=None,
        description=(
            "🎵 Live jazz with the Salem Jazz Collective this Saturday at The Vault Lounge! "
            "Doors open at 7pm, music starts at 8pm. $15 cover. "
            "Great cocktails, cozy atmosphere, perfect for date night. "
            "Follow @salemsjazcollective for updates!"
        ),
    )
    results = stage.process([event])

    assert len(results) == 1
    result = results[0]
    assert not result.metadata.get("llm_extraction_failed"), (
        f"Extraction failed: {result.metadata}"
    )
    assert len(result.tags) >= 5
    assert result.summary is not None and len(result.summary) > 0


@pytest.mark.slow
def test_bypass_not_called_with_prepopulated_tags():
    """Confirm Ollama is not called when event already has tags."""
    from src.utils.ollama_client import OllamaClient
    from src.processing.extraction import OllamaExtractionProvider
    from src.processing.extraction_stage import ExtractionStage

    # Use a mock client that will raise if called
    client = MagicMock()
    client.chat.side_effect = AssertionError("Ollama should not be called when tags exist")

    provider = OllamaExtractionProvider(client=client, model="gemma4:e4b", min_tags=5)
    stage = ExtractionStage(provider=provider, image_fetcher=None, logger=_make_logger())

    event = _make_event(
        tags=["jazz", "live music", "evening", "venue", "weekend"],
        summary="Pre-existing summary.",
    )
    results = stage.process([event])

    # If we got here, Ollama was not called
    assert results[0].tags == ["jazz", "live music", "evening", "venue", "weekend"]
    assert results[0].summary == "Pre-existing summary."
