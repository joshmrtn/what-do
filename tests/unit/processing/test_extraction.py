"""Unit tests for ExtractionProvider and OllamaExtractionProvider."""

from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock
import json

import pytest

from src.models.tag import Tag
from src.processing.extraction import (
    ExtractionError,
    ExtractionProvider,
    ExtractionResult,
    OllamaExtractionProvider,
)


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
# Reference date grounding
# ---------------------------------------------------------------------------


def test_reference_date_injected_into_prompt():

    client = _make_client(_valid_response())
    provider = OllamaExtractionProvider(client=client, min_tags=5)
    provider.extract("some event text", reference_date=datetime(2026, 8, 3))

    prompt = client.chat.call_args.kwargs["messages"][0]["content"]
    assert "2026-08-03" in prompt


def test_no_reference_date_omits_date_context():

    client = _make_client(_valid_response())
    provider = OllamaExtractionProvider(client=client, min_tags=5)
    provider.extract("some event text")

    prompt = client.chat.call_args.kwargs["messages"][0]["content"]
    assert "Today's date" not in prompt


# ---------------------------------------------------------------------------
# ExtractionProvider ABC
# ---------------------------------------------------------------------------


def test_mock_provider_satisfies_abc():

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

    client = _make_client(_valid_response())
    provider = OllamaExtractionProvider(client=client, model="gemma4:e4b", min_tags=5)

    result = provider.extract("Come hear jazz tonight at The Vault Lounge starting at 8pm!")

    assert result.title == "Jazz Night at The Vault"
    assert result.venue == "The Vault Lounge"
    assert result.start_time is not None
    assert len(result.tags) == 5
    assert result.summary == "A live jazz performance at The Vault Lounge on Sunday evening."


def test_null_optional_fields_are_none():

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

    client = _make_client(_valid_response())
    provider = OllamaExtractionProvider(client=client, model="gemma4:e4b", min_tags=5)

    result = provider.extract("Jazz night at 8pm")
    assert isinstance(result.start_time, datetime)


# ---------------------------------------------------------------------------
# Schema enforcement — invalid outputs trigger ExtractionError + retry
# ---------------------------------------------------------------------------


def test_fewer_than_min_tags_raises_after_retry():

    # Both calls return only 3 tags
    client = _make_client(_valid_response(tags=["a", "b", "c"]))
    provider = OllamaExtractionProvider(client=client, model="gemma4:e4b", min_tags=5)

    with pytest.raises(ExtractionError, match="tag"):
        provider.extract("some event text")

    assert client.chat.call_count == 2


def test_missing_summary_raises_after_retry():

    client = _make_client(_valid_response(include_summary=False))
    provider = OllamaExtractionProvider(client=client, model="gemma4:e4b", min_tags=5)

    with pytest.raises(ExtractionError, match="summary"):
        provider.extract("some event text")

    assert client.chat.call_count == 2


def test_invalid_json_raises_after_retry():

    client = _make_client("Here is the event info: Jazz night tonight, it should be fun!")
    provider = OllamaExtractionProvider(client=client, model="gemma4:e4b", min_tags=5)

    with pytest.raises(ExtractionError, match="JSON"):
        provider.extract("some event text")

    assert client.chat.call_count == 2


def test_retry_succeeds_on_second_attempt():

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

    # With min_tags=3, a 3-tag response should succeed
    client = _make_client(_valid_response(tags=["a", "b", "c"]))
    provider = OllamaExtractionProvider(client=client, model="gemma4:e4b", min_tags=3)

    result = provider.extract("some event")
    assert len(result.tags) == 3


# ---------------------------------------------------------------------------
# Image handling
# ---------------------------------------------------------------------------


def test_image_bytes_passed_to_client():

    client = _make_client(_valid_response())
    provider = OllamaExtractionProvider(client=client, model="gemma4:e4b", min_tags=5)

    provider.extract("event text", image_bytes=b"\x89PNG fake image data")

    call_kwargs = client.chat.call_args[1]
    assert "images" in call_kwargs
    assert len(call_kwargs["images"]) == 1
    assert call_kwargs["images"][0] == b"\x89PNG fake image data"


def test_no_image_bytes_no_images_kwarg():

    client = _make_client(_valid_response())
    provider = OllamaExtractionProvider(client=client, model="gemma4:e4b", min_tags=5)

    provider.extract("event text", image_bytes=None)

    call_kwargs = client.chat.call_args[1]
    assert "images" not in call_kwargs or call_kwargs.get("images") is None


# ---------------------------------------------------------------------------
# Weighted tags (centrality)
# ---------------------------------------------------------------------------


def _weighted(pairs: list[tuple[str, float]]) -> str:
    """Build a response whose tags carry centrality weights."""
    payload = {
        "title": "Punk Rock Night",
        "venue": "The Dive",
        "start_time": "2026-06-22T21:00:00",
        "end_time": None,
        "tags": [{"tag": t, "weight": w} for t, w in pairs],
        "summary": "Three local punk bands play a $5 show.",
    }
    return json.dumps(payload)


_FIVE_WEIGHTED = [
    ("punk rock", 1.0), ("live music", 0.9), ("local bands", 0.7),
    ("all ages", 0.5), ("bar", 0.1),
]


def test_weighted_tags_preserve_weights():

    client = _make_client(_weighted(_FIVE_WEIGHTED))
    provider = OllamaExtractionProvider(client=client, min_tags=5)

    result = provider.extract("punk show tonight")

    assert result.tags == [Tag(text=t, weight=w) for t, w in _FIVE_WEIGHTED]


def test_bare_string_tags_default_to_full_weight():
    """Models drift; a plain string list must not fail extraction."""

    client = _make_client(_valid_response(tags=["jazz", "live music", "venue", "evening", "lounge"]))
    provider = OllamaExtractionProvider(client=client, min_tags=5)

    result = provider.extract("jazz tonight")

    assert result.tags[0] == Tag(text="jazz", weight=1.0)
    assert all(t.weight == 1.0 for t in result.tags)


def test_missing_weight_defaults_to_full_weight():

    payload = json.loads(_weighted(_FIVE_WEIGHTED))
    del payload["tags"][2]["weight"]
    client = _make_client(json.dumps(payload))
    provider = OllamaExtractionProvider(client=client, min_tags=5)

    result = provider.extract("punk show tonight")

    assert result.tags[2].weight == 1.0


def test_non_numeric_weight_defaults_to_full_weight():

    payload = json.loads(_weighted(_FIVE_WEIGHTED))
    payload["tags"][1]["weight"] = "very important"
    client = _make_client(json.dumps(payload))
    provider = OllamaExtractionProvider(client=client, min_tags=5)

    result = provider.extract("punk show tonight")

    assert result.tags[1].weight == 1.0


@pytest.mark.parametrize("raw_weight,expected", [(1.7, 1.0), (-0.4, 0.0)])
def test_out_of_range_weight_is_clamped(raw_weight, expected):

    payload = json.loads(_weighted(_FIVE_WEIGHTED))
    payload["tags"][0]["weight"] = raw_weight
    client = _make_client(json.dumps(payload))
    provider = OllamaExtractionProvider(client=client, min_tags=5)

    result = provider.extract("punk show tonight")

    assert result.tags[0].weight == expected


def test_min_tag_count_enforced_on_weighted_tags():

    client = _make_client(_weighted(_FIVE_WEIGHTED[:3]))
    provider = OllamaExtractionProvider(client=client, min_tags=5)

    with pytest.raises(ExtractionError, match="tag"):
        provider.extract("punk show tonight")


def test_tag_entry_without_text_is_skipped():

    payload = json.loads(_weighted(_FIVE_WEIGHTED + [("", 0.5)]))
    client = _make_client(json.dumps(payload))
    provider = OllamaExtractionProvider(client=client, min_tags=5)

    result = provider.extract("punk show tonight")

    assert len(result.tags) == 5
    assert all(t.text for t in result.tags)


def test_prompt_requests_centrality_weights():

    client = _make_client(_weighted(_FIVE_WEIGHTED))
    provider = OllamaExtractionProvider(client=client, min_tags=5)
    provider.extract("punk show tonight")

    prompt = client.chat.call_args.kwargs["messages"][0]["content"]
    assert "weight" in prompt.lower()
    assert "central" in prompt.lower()


def test_emoji_stripped_from_tag_text():
    """Emoji embed to a constant vector and dilute the words beside them."""

    pairs = [("🎤 karaoke", 1.0), ("live music 🎸", 0.9), ("punk rock", 0.7),
             ("all ages", 0.5), ("bar", 0.1)]
    client = _make_client(_weighted(pairs))
    provider = OllamaExtractionProvider(client=client, min_tags=5)

    result = provider.extract("punk show tonight")

    assert result.tags[0] == Tag(text="karaoke", weight=1.0)
    assert result.tags[1] == Tag(text="live music", weight=0.9)


def test_emoji_only_tag_falls_back_to_its_name():
    """Nothing to dilute, so the name carries more signal than a dropped tag."""

    pairs = _FIVE_WEIGHTED + [("🎤", 0.4)]
    client = _make_client(_weighted(pairs))
    provider = OllamaExtractionProvider(client=client, min_tags=6)

    result = provider.extract("punk show tonight")

    assert result.tags[-1] == Tag(text="microphone", weight=0.4)


def test_tag_of_only_invisible_characters_is_dropped():

    pairs = _FIVE_WEIGHTED + [("​﻿", 0.4)]
    client = _make_client(_weighted(pairs))
    provider = OllamaExtractionProvider(client=client, min_tags=6)

    with pytest.raises(ExtractionError, match="tag count 5"):
        provider.extract("punk show tonight")


# ---------------------------------------------------------------------------
# setting: indoor / outdoor / unknown
# ---------------------------------------------------------------------------


_OMIT = object()


def _response_with_setting(setting=_OMIT) -> str:
    payload = json.loads(_valid_response())
    if setting is not _OMIT:
        payload["setting"] = setting
    return json.dumps(payload)


def _extract_setting(setting=_OMIT) -> str:

    client = _make_client(_response_with_setting(setting))
    return OllamaExtractionProvider(client=client, min_tags=5).extract("text").setting


@pytest.mark.parametrize("value", ["indoor", "outdoor", "unknown"])
def test_setting_parsed_from_each_allowed_value(value):
    assert _extract_setting(value) == value


def test_setting_is_case_and_space_insensitive():
    assert _extract_setting("  Outdoor ") == "outdoor"


@pytest.mark.parametrize("value", ["outside", "", None, 7, ["outdoor"]])
def test_off_enum_setting_coerces_to_unknown(value):
    """A bad enum is not worth a 3-minute local LLM retry — degrade, do not fail."""
    assert _extract_setting(value) == "unknown"


def test_missing_setting_key_coerces_to_unknown():
    assert _extract_setting() == "unknown"


def test_off_enum_setting_does_not_trigger_a_retry():

    client = _make_client(_response_with_setting("outside"))
    OllamaExtractionProvider(client=client, min_tags=5).extract("text")
    assert client.chat.call_count == 1


def test_prompt_requests_the_setting_field():

    client = _make_client(_valid_response())
    OllamaExtractionProvider(client=client, min_tags=5).extract("text")
    prompt = client.chat.call_args.kwargs["messages"][0]["content"]
    assert "setting" in prompt
    assert "outdoor" in prompt
