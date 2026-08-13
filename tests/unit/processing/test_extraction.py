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




def _provider(min_tags: int = 1, tags=None) -> OllamaExtractionProvider:
    """A provider whose model returns exactly the tags a test names."""
    payload = {
        "title": None,
        "venue": None,
        "start_time": None,
        "end_time": None,
        "tags": tags if tags is not None else [{"tag": "music", "weight": 1.0}],
        "summary": "A live music performance.",
        "setting": "unknown",
    }
    client = _make_client(json.dumps(payload))
    return OllamaExtractionProvider(client=client, min_tags=min_tags)


def _sent_prompt(provider: OllamaExtractionProvider) -> str:
    return provider._client.chat.call_args.kwargs["messages"][0]["content"]


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
# Provenance — what actually answered
# ---------------------------------------------------------------------------


def test_result_names_the_model_that_answered():
    """Configured and actual can differ, and the refit needs the actual one."""
    client = _make_client(_valid_response())
    provider = OllamaExtractionProvider(client=client, model="gemma4:e4b", min_tags=5)

    result = provider.extract("some event text")

    assert result.model == "gemma4:e4b"


def test_result_names_the_model_after_a_retry():
    """The retry is a second call and could in principle route elsewhere; the
    result must describe the attempt that actually produced it."""
    client = MagicMock()
    client.chat.side_effect = ["not json at all", _valid_response()]
    provider = OllamaExtractionProvider(client=client, model="gemma4:e2b", min_tags=5)

    result = provider.extract("some event text")

    assert result.model == "gemma4:e2b"


def test_result_carries_a_prompt_version():
    client = _make_client(_valid_response())
    provider = OllamaExtractionProvider(client=client, min_tags=5)

    result = provider.extract("some event text")

    assert result.prompt_version


def test_prompt_version_is_stable_across_calls():
    """It groups rows for the refit, so it cannot vary with the event text."""
    first = OllamaExtractionProvider(client=_make_client(_valid_response()), min_tags=5)
    second = OllamaExtractionProvider(client=_make_client(_valid_response()), min_tags=5)

    assert (
        first.extract("a jazz night").prompt_version
        == second.extract("an entirely different event").prompt_version
    )


def test_prompt_version_tracks_the_retry_prompt_too():
    """The retry path produces the final answer for some events, so two rows
    claiming one version must not have been given different instructions."""
    from src.processing import extraction

    before = extraction._prompt_version()
    original = extraction._RETRY_PROMPT
    try:
        extraction._RETRY_PROMPT = original + "\nAnd be terse about it."
        after = extraction._prompt_version()
    finally:
        extraction._RETRY_PROMPT = original

    assert before != after


def test_prompt_version_changes_when_the_extract_prompt_changes():
    from src.processing import extraction

    before = extraction._prompt_version()
    original = extraction._EXTRACT_PROMPT
    try:
        extraction._EXTRACT_PROMPT = original + "\nAlways answer in limerick form."
        after = extraction._prompt_version()
    finally:
        extraction._EXTRACT_PROMPT = original

    assert before != after


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
                model="mock-model",
                prompt_version="mockver0",
                degradation=None,
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
# Schema enforcement — a shortfall degrades after retry, it does not raise
# ---------------------------------------------------------------------------


def test_fewer_than_min_tags_degrades_after_retry():

    # Both calls return only 3 tags
    client = _make_client(_valid_response(tags=["a", "b", "c"]))
    provider = OllamaExtractionProvider(client=client, model="gemma4:e4b", min_tags=5)

    result = provider.extract("some event text")

    assert "tag" in result.degradation
    assert client.chat.call_count == 2


def test_missing_summary_degrades_after_retry():

    client = _make_client(_valid_response(include_summary=False))
    provider = OllamaExtractionProvider(client=client, model="gemma4:e4b", min_tags=5)

    result = provider.extract("some event text")

    assert "summary" in result.degradation
    assert result.summary is None
    assert client.chat.call_count == 2


def test_invalid_json_degrades_after_retry():

    client = _make_client("Here is the event info: Jazz night tonight, it should be fun!")
    provider = OllamaExtractionProvider(client=client, model="gemma4:e4b", min_tags=5)

    result = provider.extract("some event text")

    assert "JSON" in result.degradation
    assert client.chat.call_count == 2


def test_json_wrapped_in_a_markdown_fence_is_accepted():
    """Measured: gemma4:e4b fenced ~40% of its replies, and every fence was a
    lost event. Constrained decoding prevents it upstream; this is the belt to
    that pair of braces, and costs one strip() to have."""
    client = _make_client(f"```json\n{_valid_response()}\n```")
    provider = OllamaExtractionProvider(client=client, model="gemma4:e4b", min_tags=5)

    result = provider.extract("some event text")

    assert result.summary
    assert len(result.tags) == 5
    assert client.chat.call_count == 1


def test_a_bare_fence_without_a_language_tag_is_accepted():
    client = _make_client(f"```\n{_valid_response()}\n```")
    provider = OllamaExtractionProvider(client=client, model="gemma4:e4b", min_tags=5)

    assert provider.extract("some event text").summary
    assert client.chat.call_count == 1


def test_a_fence_with_surrounding_chatter_is_accepted():
    client = _make_client(f"Here you go:\n\n```json\n{_valid_response()}\n```\n\nHope that helps!")
    provider = OllamaExtractionProvider(client=client, model="gemma4:e4b", min_tags=5)

    assert provider.extract("some event text").summary
    assert client.chat.call_count == 1


def test_prose_with_no_json_at_all_still_fails():
    """Stripping a fence must not become 'find something vaguely brace-shaped'."""
    client = _make_client("Jazz night tonight, it should be fun!")
    provider = OllamaExtractionProvider(client=client, model="gemma4:e4b", min_tags=5)

    result = provider.extract("some event text")

    assert "JSON" in result.degradation
    assert result.tags == []
    assert result.summary is None


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

    result = provider.extract("punk show tonight")

    assert "tag" in result.degradation


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

    result = provider.extract("punk show tonight")

    assert "tag count 5" in result.degradation
    assert len(result.tags) == 5


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


class TestTagsAreNotPadded:
    """The floor was the primary defect, measured 2026-08-11.

    `extraction.py` returned a sub-minimum result as a *failure*, so the system
    retried until it got padding. At five, gemma answered `Ron & Sheila Schrank`
    with `event 0.1 | scheduling 0.1 | information 0.1 | null_data 0.1 |
    missing_text 0.5` — commentary on its own confusion, embedded and scored.
    """

    def test_a_single_honest_tag_is_accepted(self):
        provider = _provider(min_tags=1, tags=[{"tag": "music", "weight": 1.0}])

        result = provider.extract("Fred Ellsworth")

        assert [t.text for t in result.tags] == ["music"]

    def test_the_prompt_states_there_is_no_minimum(self):
        provider = _provider(min_tags=1)
        provider.extract("Fred Ellsworth")

        prompt = _sent_prompt(provider)
        assert "at least" not in prompt.lower()
        assert "no minimum" in prompt.lower()

    def test_the_prompt_forbids_inferring_style_from_a_name(self):
        provider = _provider(min_tags=1)
        provider.extract("Fred Ellsworth")

        # The prompt is wrapped, so compare on collapsed whitespace.
        prompt = " ".join(_sent_prompt(provider).lower().split())
        assert "a performer's proper name does not name a genre" in prompt
        assert "if the text does not name the style, emit no style tag" in prompt

    def test_the_prompt_says_what_an_event_category_is(self):
        """Without this the model reads a section heading as event copy."""
        provider = _provider(min_tags=1)
        provider.extract("Fred Ellsworth\nEvent category: Music")

        prompt = " ".join(_sent_prompt(provider).split())
        assert "section heading from a listing site" in prompt


class TestPlaceholderTagsAreRejected:
    """Tags that carry no meaning of their own still get embedded and scored."""

    def test_a_placeholder_tag_is_dropped(self):
        provider = _provider(
            min_tags=1,
            tags=[
                {"tag": "live music", "weight": 1.0},
                {"tag": "genre", "weight": 0.5},
                {"tag": "music genre", "weight": 0.4},
                {"tag": "artist", "weight": 0.3},
            ],
        )

        result = provider.extract("Tyler Bard")

        assert [t.text for t in result.tags] == ["live music"]

    def test_a_tag_that_merely_echoes_the_title_is_dropped(self):
        provider = _provider(
            min_tags=1,
            tags=[{"tag": "lee hawkins", "weight": 0.9}, {"tag": "music", "weight": 1.0}],
        )

        result = provider.extract("Lee Hawkins")

        assert [t.text for t in result.tags] == ["music"]

    def test_a_real_tag_that_appears_in_the_title_survives(self):
        """`Jazz Night` -> `jazz` is the case that makes this rule non-trivial."""
        provider = _provider(
            min_tags=1,
            tags=[{"tag": "jazz", "weight": 1.0}, {"tag": "music", "weight": 0.5}],
        )

        result = provider.extract("Jazz Night")

        assert [t.text for t in result.tags] == ["jazz", "music"]

    def test_dropping_every_tag_degrades_rather_than_failing(self):
        provider = _provider(min_tags=1, tags=[{"tag": "genre", "weight": 0.5}])

        result = provider.extract("Tyler Bard")

        assert result.tags == []
        assert "tag count 0" in result.degradation


class TestDegradationIsRecordedNotRaised:
    """A shortfall is graded, not discarded.

    `tag_confidence` already pulls thin evidence toward mid-ranking; refusing
    the result outright was a second, cruder judgement on top of a working one.
    The measured cost was ten events a night ranking on tags from a prompt
    version we had rejected, invisibly, because a failure wrote nothing at all.
    """

    def test_a_clean_extraction_records_no_degradation(self):
        provider = _provider(min_tags=1)

        assert provider.extract("Jazz Night").degradation is None

    def test_a_degraded_result_still_carries_its_provenance(self):
        """The refit's dataset is exactly the rows a failure used to omit, and
        they are the shortest inputs — the region the curve is most sensitive
        in. A degraded row with no model on it is a row the refit cannot use."""
        client = _make_client(_valid_response(tags=[]))
        provider = OllamaExtractionProvider(client=client, model="gemma4:e4b", min_tags=1)

        result = provider.extract("Limón Dance Company")

        assert result.model == "gemma4:e4b"
        assert result.prompt_version

    def test_the_usable_half_of_a_partial_reply_survives(self):
        """`Stand-up Comedy` returned a real summary and one true tag; the echo
        rule ate the tag and the whole reply went with it. Whatever the model
        did get right is kept."""
        client = _make_client(
            json.dumps(
                {
                    "title": "Stand-up Comedy",
                    "venue": None,
                    "start_time": None,
                    "end_time": None,
                    "tags": [{"tag": "stand-up comedy", "weight": 1.0}],
                    "summary": "This is a stand-up comedy event.",
                    "setting": "unknown",
                }
            )
        )
        provider = OllamaExtractionProvider(client=client, min_tags=1)

        result = provider.extract("Stand-up Comedy")

        assert result.tags == []
        assert result.summary == "This is a stand-up comedy event."
        assert result.title == "Stand-up Comedy"
        assert "tag count 0" in result.degradation

    def test_every_shortfall_in_one_reply_is_named(self):
        """The all-null reply is short of both, and a counter that only ever
        sees the first reason cannot tell the two failure shapes apart."""
        client = _make_client(
            json.dumps(
                {
                    "title": None,
                    "venue": None,
                    "start_time": None,
                    "end_time": None,
                    "tags": [],
                    "summary": None,
                    "setting": None,
                }
            )
        )
        provider = OllamaExtractionProvider(client=client, min_tags=1)

        result = provider.extract("Limón Dance Company")

        assert "tag count 0" in result.degradation
        assert "summary" in result.degradation

    def test_a_provider_that_cannot_answer_at_all_still_raises(self):
        """`ExtractionError` stops meaning 'the reply was thin' and keeps
        meaning 'there was no reply' — the case that must not write a hash,
        because unlike a shortfall it may well succeed tomorrow."""
        client = MagicMock()
        client.chat.side_effect = ExtractionError("model unavailable")
        provider = OllamaExtractionProvider(client=client, min_tags=1)

        with pytest.raises(ExtractionError):
            provider.extract("Jazz Night")
