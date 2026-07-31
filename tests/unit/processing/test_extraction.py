"""Unit tests for ExtractionProvider and OllamaExtractionProvider."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest


def _make_client(response_text: str):
    client = MagicMock()
    client.chat.return_value = response_text
    return client


def _valid_response(tags: list[str] | None = None, include_summary: bool = True) -> str:
    payload = {
        "title": "Jazz Night at The Vault",
        "venue": "The Vault Lounge",
        "start_time": "2026-06-22T20:00:00",
        "end_time": "2026-06-22T23:00:00",
        "tags": tags if tags is not None else ["jazz", "live music", "venue", "evening", "lounge"],
        "summary": "A live jazz performance at The Vault Lounge on Sunday evening." if include_summary else None,
    }
    if not include_summary:
        del payload["summary"]
    return json.dumps(payload)


# ---------------------------------------------------------------------------
# ExtractionProvider ABC
# ---------------------------------------------------------------------------


def test_mock_provider_satisfies_abc():
    from src.processing.extraction import ExtractionProvider, ExtractionResult

    class MockProvider(ExtractionProvider):
        def extract(self, text: str, image_bytes: bytes | None = None) -> ExtractionResult:
            return ExtractionResult(
                title="test",
                venue=None,
                start_time=None,
                end_time=None,
                tags=["a", "b", "c", "d", "e"],
                summary="A test event.",
            )

    provider = MockProvider()
    result = provider.extract("some text")
    assert result.tags == ["a", "b", "c", "d", "e"]


# ---------------------------------------------------------------------------
# OllamaExtractionProvider — valid output
# ---------------------------------------------------------------------------


def test_valid_response_parsed_correctly():
    from src.processing.extraction import OllamaExtractionProvider

    client = _make_client(_valid_response())
    provider = OllamaExtractionProvider(client=client, model="gemma4:e4b", min_tags=5)

    result = provider.extract("Come hear jazz tonight at The Vault Lounge starting at 8pm!")

    assert result.title == "Jazz Night at The Vault"
    assert result.venue == "The Vault Lounge"
    assert result.start_time is not None
    assert len(result.tags) == 5
    assert result.summary == "A live jazz performance at The Vault Lounge on Sunday evening."


def test_null_optional_fields_are_none():
    from src.processing.extraction import OllamaExtractionProvider

    payload = {
        "title": None,
        "venue": None,
        "start_time": None,
        "end_time": None,
        "tags": ["tag1", "tag2", "tag3", "tag4", "tag5"],
        "summary": "An event with minimal info.",
    }
    client = _make_client(json.dumps(payload))
    provider = OllamaExtractionProvider(client=client, model="gemma4:e4b", min_tags=5)

    result = provider.extract("vague post")
    assert result.title is None
    assert result.venue is None
    assert result.start_time is None


def test_start_time_parsed_as_datetime():
    from src.processing.extraction import OllamaExtractionProvider

    client = _make_client(_valid_response())
    provider = OllamaExtractionProvider(client=client, model="gemma4:e4b", min_tags=5)

    result = provider.extract("Jazz night at 8pm")
    from datetime import datetime
    assert isinstance(result.start_time, datetime)


# ---------------------------------------------------------------------------
# Schema enforcement — invalid outputs trigger ExtractionError + retry
# ---------------------------------------------------------------------------


def test_fewer_than_min_tags_raises_after_retry():
    from src.processing.extraction import ExtractionError, OllamaExtractionProvider

    # Both calls return only 3 tags
    client = _make_client(_valid_response(tags=["a", "b", "c"]))
    provider = OllamaExtractionProvider(client=client, model="gemma4:e4b", min_tags=5)

    with pytest.raises(ExtractionError, match="tag"):
        provider.extract("some event text")

    assert client.chat.call_count == 2


def test_missing_summary_raises_after_retry():
    from src.processing.extraction import ExtractionError, OllamaExtractionProvider

    client = _make_client(_valid_response(include_summary=False))
    provider = OllamaExtractionProvider(client=client, model="gemma4:e4b", min_tags=5)

    with pytest.raises(ExtractionError, match="summary"):
        provider.extract("some event text")

    assert client.chat.call_count == 2


def test_invalid_json_raises_after_retry():
    from src.processing.extraction import ExtractionError, OllamaExtractionProvider

    client = _make_client("Here is the event info: Jazz night tonight, it should be fun!")
    provider = OllamaExtractionProvider(client=client, model="gemma4:e4b", min_tags=5)

    with pytest.raises(ExtractionError, match="JSON"):
        provider.extract("some event text")

    assert client.chat.call_count == 2


def test_retry_succeeds_on_second_attempt():
    from src.processing.extraction import OllamaExtractionProvider

    client = MagicMock()
    client.chat.side_effect = [
        "Oops I forgot to format that as JSON",  # first: invalid
        _valid_response(),                         # second: valid
    ]
    provider = OllamaExtractionProvider(client=client, model="gemma4:e4b", min_tags=5)

    result = provider.extract("Jazz night at The Vault")
    assert result.tags is not None
    assert len(result.tags) == 5
    assert client.chat.call_count == 2


def test_min_tags_configurable():
    from src.processing.extraction import OllamaExtractionProvider

    # With min_tags=3, a 3-tag response should succeed
    client = _make_client(_valid_response(tags=["a", "b", "c"]))
    provider = OllamaExtractionProvider(client=client, model="gemma4:e4b", min_tags=3)

    result = provider.extract("some event")
    assert len(result.tags) == 3


# ---------------------------------------------------------------------------
# Image handling
# ---------------------------------------------------------------------------


def test_image_bytes_passed_to_client():
    from src.processing.extraction import OllamaExtractionProvider

    client = _make_client(_valid_response())
    provider = OllamaExtractionProvider(client=client, model="gemma4:e4b", min_tags=5)

    provider.extract("event text", image_bytes=b"\x89PNG fake image data")

    call_kwargs = client.chat.call_args[1]
    assert "images" in call_kwargs
    assert len(call_kwargs["images"]) == 1
    assert call_kwargs["images"][0] == b"\x89PNG fake image data"


def test_no_image_bytes_no_images_kwarg():
    from src.processing.extraction import OllamaExtractionProvider

    client = _make_client(_valid_response())
    provider = OllamaExtractionProvider(client=client, model="gemma4:e4b", min_tags=5)

    provider.extract("event text", image_bytes=None)

    call_kwargs = client.chat.call_args[1]
    assert "images" not in call_kwargs or call_kwargs.get("images") is None
