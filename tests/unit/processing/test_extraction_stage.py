"""Unit tests for ExtractionStage."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock, call
import io

import pytest

from src.models.event import Event
from src.models.source_type import SYNTHETIC
from src.models.tag import Tag
from src.processing.extraction import ExtractionError, ExtractionResult, OllamaExtractionProvider
from src.processing.extraction_stage import (
    ExtractionStage,
    extraction_input,
    extraction_input_hash,
)
from src.processing.image_fetcher import ImageFetchError
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


def _make_provider(tags=None, summary="A great event.", setting="unknown"):

    provider = MagicMock()
    provider.extract.return_value = ExtractionResult(
        title="Extracted Title",
        venue="Extracted Venue",
        start_time=datetime(2026, 6, 22, 20, 0, 0),
        end_time=None,
        tags=tags or [Tag(text=t) for t in
                      ["jazz", "live music", "evening", "venue", "weekend"]],
        summary=summary,
        setting=setting,
    )
    return provider


def _make_fetcher(content: bytes = b"fake image bytes"):
    fetcher = MagicMock()
    fetcher.fetch.return_value = content
    return fetcher


def test_reference_date_from_get_now_passed_to_provider():

    provider = _make_provider()
    fixed = datetime(2026, 8, 3, 9, 0, 0, tzinfo=timezone.utc)
    stage = ExtractionStage(
        provider=provider,
        image_fetcher=None,
        logger=_make_logger(),
        get_now=lambda: fixed,
    )
    stage.process([_make_event(tags=[])])

    assert provider.extract.call_args.kwargs["reference_date"] == fixed


# ---------------------------------------------------------------------------
# Bypass
# ---------------------------------------------------------------------------


def test_bypass_when_the_input_is_unchanged():

    event = _make_event(tags=[Tag(text=t) for t in
                              ["jazz", "live", "music", "fun", "night"]])
    event.extraction_input_hash = extraction_input_hash(event)
    provider = _make_provider()
    stage = ExtractionStage(provider=provider, image_fetcher=None, logger=_make_logger())

    results = stage.process([event])

    provider.extract.assert_not_called()
    assert results[0].tags == [Tag(text=t) for t in
                               ["jazz", "live", "music", "fun", "night"]]


def test_tags_without_a_hash_are_re_extracted():
    """Tags with no record of what produced them are not evidence of a finished run.

    Only a synthetic event legitimately carries tags it did not extract, and
    that is settled by provenance rather than by this check.
    """
    event = _make_event(tags=[Tag(text="stale")])
    provider = _make_provider()
    stage = ExtractionStage(provider=provider, image_fetcher=None, logger=_make_logger())

    stage.process([event])

    provider.extract.assert_called_once()


def test_extraction_called_when_tags_empty():

    event = _make_event(tags=[])
    provider = _make_provider()
    stage = ExtractionStage(provider=provider, image_fetcher=None, logger=_make_logger())

    stage.process([event])

    provider.extract.assert_called_once()


# ---------------------------------------------------------------------------
# Field merging
# ---------------------------------------------------------------------------


def test_tags_and_summary_always_written():

    event = _make_event()
    provider = _make_provider(tags=[Tag(text=c) for c in "abcde"], summary="Excellent event.")
    stage = ExtractionStage(provider=provider, image_fetcher=None, logger=_make_logger())

    results = stage.process([event])

    assert results[0].tags == [Tag(text=c) for c in "abcde"]
    assert results[0].summary == "Excellent event."


def test_llm_fills_null_title():

    event = _make_event(title=None)
    provider = _make_provider()
    stage = ExtractionStage(provider=provider, image_fetcher=None, logger=_make_logger())

    results = stage.process([event])

    assert results[0].title == "Extracted Title"


def test_llm_does_not_overwrite_existing_title():

    event = _make_event(title="Existing Title From Normalization")
    provider = _make_provider()
    stage = ExtractionStage(provider=provider, image_fetcher=None, logger=_make_logger())

    results = stage.process([event])

    assert results[0].title == "Existing Title From Normalization"


def test_llm_fills_null_venue():

    event = _make_event(venue=None)
    provider = _make_provider()
    stage = ExtractionStage(provider=provider, image_fetcher=None, logger=_make_logger())

    results = stage.process([event])

    assert results[0].venue == "Extracted Venue"


def test_llm_does_not_overwrite_existing_venue():

    event = _make_event(venue="The Vault Lounge")
    provider = _make_provider()
    stage = ExtractionStage(provider=provider, image_fetcher=None, logger=_make_logger())

    results = stage.process([event])

    assert results[0].venue == "The Vault Lounge"


def test_llm_fills_null_start_time():

    event = _make_event(start_time=None)
    provider = _make_provider()
    stage = ExtractionStage(provider=provider, image_fetcher=None, logger=_make_logger())

    results = stage.process([event])

    assert results[0].start_time == datetime(2026, 6, 22, 20, 0, 0)


def test_llm_does_not_overwrite_existing_start_time():

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

    event = _make_event(image_url=None)
    fetcher = _make_fetcher()
    provider = _make_provider()
    stage = ExtractionStage(provider=provider, image_fetcher=fetcher, logger=_make_logger())

    stage.process([event])

    fetcher.fetch.assert_not_called()
    _, kwargs = provider.extract.call_args
    assert kwargs.get("image_bytes") is None


def test_image_fetch_error_logs_warning_and_continues():

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

    provider = MagicMock()
    provider.extract.side_effect = ExtractionError("LLM confused")
    stage = ExtractionStage(provider=provider, image_fetcher=None, logger=_make_logger())

    event = _make_event()
    results = stage.process([event])

    assert results[0].metadata.get("llm_extraction_failed") is True
    assert results[0].tags == []


def test_one_failed_event_does_not_stop_others():

    provider = MagicMock()
    provider.extract.side_effect = [
        ExtractionError("bad"),
        provider.extract.return_value,  # won't reach, side_effect overrides
    ]

    good_result_mock = MagicMock()
    good_result_mock.tags = [Tag(text=c) for c in "abcde"]
    good_result_mock.summary = "Good event."
    good_result_mock.title = None
    good_result_mock.venue = None
    good_result_mock.start_time = None
    good_result_mock.end_time = None


    provider2 = MagicMock()
    provider2.extract.side_effect = [
        ExtractionError("bad"),
        ExtractionResult(
            title=None, venue=None, start_time=None, end_time=None,
            tags=[Tag(text=c) for c in "abcde"], summary="Good."
        ),
    ]

    event1 = _make_event(event_id="evt-fail")
    event2 = _make_event(event_id="evt-ok")

    stage = ExtractionStage(provider=provider2, image_fetcher=None, logger=_make_logger())
    results = stage.process([event1, event2])

    assert len(results) == 2
    assert results[0].metadata.get("llm_extraction_failed") is True
    assert results[1].tags == [Tag(text=c) for c in "abcde"]


# ---------------------------------------------------------------------------
# Input text construction
# ---------------------------------------------------------------------------


def test_input_text_combines_title_and_description():

    event = _make_event(title="Jazz Night", description="Live jazz at the waterfront.")
    provider = _make_provider()
    stage = ExtractionStage(provider=provider, image_fetcher=None, logger=_make_logger())

    stage.process([event])

    text_arg = provider.extract.call_args[0][0]
    assert "Jazz Night" in text_arg
    assert "Live jazz at the waterfront." in text_arg


def test_bypass_not_called_when_the_input_is_unchanged():
    """Confirm Ollama is not reached when the stored hash still matches."""

    # A client that raises if it is called at all.
    client = MagicMock()
    client.chat.side_effect = AssertionError("Ollama should not be called when done")

    provider = OllamaExtractionProvider(client=client, model="gemma4:e4b", min_tags=5)
    stage = ExtractionStage(provider=provider, image_fetcher=None, logger=_make_logger())

    event = _make_event(
        tags=[Tag(text=t) for t in ["jazz", "live music", "evening", "venue", "weekend"]],
        summary="Pre-existing summary.",
    )
    event.extraction_input_hash = extraction_input_hash(event)
    results = stage.process([event])

    # If we got here, Ollama was not called
    assert [t.text for t in results[0].tags] == [
        "jazz", "live music", "evening", "venue", "weekend"
    ]
    assert results[0].summary == "Pre-existing summary."


# ---------------------------------------------------------------------------
# setting
# ---------------------------------------------------------------------------


def test_extracted_setting_is_applied_to_the_event():
    stage = ExtractionStage(
        provider=_make_provider(setting="outdoor"),
        image_fetcher=None,
        logger=_make_logger(),
    )
    assert stage.process([_make_event()])[0].setting == "outdoor"


def test_bypassed_event_keeps_its_own_setting():
    """A bypassed event skips extraction, so its setting must survive."""
    stage = ExtractionStage(
        provider=_make_provider(setting="outdoor"),
        image_fetcher=None,
        logger=_make_logger(),
    )
    preset = _make_event(tags=[Tag(text="jazz")], setting="indoor")
    preset.extraction_input_hash = extraction_input_hash(preset)
    assert stage.process([preset])[0].setting == "indoor"


# ---------------------------------------------------------------------------
# Input-hash skip rule
# ---------------------------------------------------------------------------


def test_an_unchanged_event_is_not_re_extracted():
    provider = _make_provider(tags=[Tag(text="jazz", weight=1.0)])
    stage = ExtractionStage(provider, None, _make_logger(), get_now=_now)
    event = _make_event()

    stage.process([event])
    stage.process([event])

    assert provider.extract.call_count == 1


def test_an_edited_description_is_re_extracted():
    """The gap the old `if event.tags` rule left: edited text was never revisited."""
    provider = _make_provider(tags=[Tag(text="jazz", weight=1.0)])
    stage = ExtractionStage(provider, None, _make_logger(), get_now=_now)
    event = _make_event()

    stage.process([event])
    event.description = "Actually it is a punk show now, same venue."
    stage.process([event])

    assert provider.extract.call_count == 2


def test_an_extraction_returning_no_tags_is_not_retried_forever():
    """Zero tags is a valid verdict, and was indistinguishable from never having run."""
    provider = MagicMock()
    provider.extract.return_value = ExtractionResult(
        title=None,
        venue=None,
        start_time=None,
        end_time=None,
        tags=[],
        summary="Nothing much to say about this one.",
    )
    stage = ExtractionStage(provider, None, _make_logger(), get_now=_now)
    event = _make_event()

    stage.process([event])
    stage.process([event])

    assert provider.extract.call_count == 1


def test_a_failed_extraction_is_retried_next_run():
    """The hash is stored only on success, so a failure stays distinguishable."""
    provider = MagicMock()
    provider.extract.side_effect = ExtractionError("model unavailable")
    stage = ExtractionStage(provider, None, _make_logger(), get_now=_now)
    event = _make_event()

    stage.process([event])
    stage.process([event])

    assert provider.extract.call_count == 2


def test_a_synthetic_event_is_never_extracted():
    """Its tags are hand-written config; extracting would overwrite the author."""
    provider = _make_provider(tags=[Tag(text="llm", weight=1.0)])
    stage = ExtractionStage(provider, None, _make_logger(), get_now=_now)
    event = _make_event(
        source_type=SYNTHETIC,
        title="Evening walk",
        description=None,
        tags=[Tag(text="walking", weight=1.0)],
    )

    stage.process([event])

    assert provider.extract.call_count == 0
    assert [t.text for t in event.tags] == ["walking"]


# ---------------------------------------------------------------------------
# Saving as it goes
# ---------------------------------------------------------------------------


class _Saves:
    """Records the event handed to each save, in order."""

    def __init__(self) -> None:
        self.events: list[Event] = []

    def __call__(self, event: Event) -> None:
        self.events.append(event)


def _stage_with_saves(saves: _Saves) -> ExtractionStage:
    return ExtractionStage(
        provider=_make_provider(tags=[Tag(text="music", weight=0.9)]),
        image_fetcher=None,
        logger=_make_logger(),
        get_now=_now,
        save_fn=saves,
    )


def test_every_extracted_event_is_saved_immediately():
    """A run killed at hour 19 of 20 must lose one event, not twenty.

    Batching existed because each save rewrote the whole corpus; saving one
    event no longer costs that, so the batch size that hid the cost can go.
    """
    saves = _Saves()
    events = [_make_event(title=f"e{i}") for i in range(6)]

    _stage_with_saves(saves).process(events)

    assert [e.title for e in saves.events] == [f"e{i}" for i in range(6)]


def test_a_checkpoint_carries_the_extraction_it_just_made():
    """Saving the event before its tags are attached would persist nothing of value."""
    saves = _Saves()

    _stage_with_saves(saves).process([_make_event(title="e1")])

    assert [t.text for t in saves.events[0].tags] == ["music"]


def test_nothing_is_saved_when_no_saver_is_given():
    """A dry run passes none, and must still extract normally."""
    events = [_make_event(title="e1")]

    result = ExtractionStage(
        provider=_make_provider(tags=[("music", 0.9)]),
        image_fetcher=None,
        logger=_make_logger(),
        get_now=_now,
    ).process(events)

    assert result[0].tags


def test_skipped_events_do_not_trigger_saves():
    """Only real model calls are worth checkpointing."""
    saves = _Saves()
    done = _make_event(title="done")
    done.extraction_input_hash = extraction_input_hash(done)

    _stage_with_saves(saves).process([done, done, done, done])

    assert saves.events == []


def test_a_failing_checkpoint_does_not_end_the_stage():
    """A failed save must not throw away the model time it was protecting."""
    saved: list[Event] = []

    def flaky(event: Event) -> None:
        saved.append(event)
        if len(saved) == 1:
            raise RuntimeError("disk full")

    stage = ExtractionStage(
        provider=_make_provider(tags=[("music", 0.9)]),
        image_fetcher=None,
        logger=_make_logger(),
        get_now=_now,
        save_fn=flaky,
    )

    result = stage.process([_make_event(title="e1"), _make_event(title="e2")])

    assert len(saved) == 2
    assert all(e.tags for e in result)


class TestAuthoredContentIsNotOverwritten:
    """Some sources state everything they know; the model can only invent past it.

    NSNO's listing publishes one line — `7:00 PM - Trivia - The James - Essex`.
    A summary composed from those fields carries all the signal there is, and
    the model's version came back "an evening of karaoke and trivia" for events
    that were only ever trivia.
    """

    def test_an_authored_summary_survives_extraction(self):
        event = _make_event(
            summary="Trivia at The James in Essex",
            metadata={"authored_summary": True},
        )
        provider = _make_provider(summary="Join us for a night of karaoke and trivia.")

        ExtractionStage(provider, None, logger=_make_logger()).process([event])

        assert event.summary == "Trivia at The James in Essex"

    def test_tags_still_come_from_the_model_when_only_the_summary_is_authored(self):
        event = _make_event(
            summary="Jazz Night at May Flower in Ipswich",
            metadata={"authored_summary": True},
        )
        provider = _make_provider(tags=[Tag(text="jazz", weight=1.0)])

        ExtractionStage(provider, None, logger=_make_logger()).process([event])

        assert [t.text for t in event.tags] == ["jazz"]

    def test_an_event_with_authored_tags_never_reaches_the_model(self):
        event = _make_event(
            summary="Trivia at The James in Essex",
            tags=[Tag(text="trivia", weight=1.0), Tag(text="quiz night", weight=0.8)],
            metadata={"authored_tags": True, "authored_summary": True},
        )
        provider = _make_provider()

        ExtractionStage(provider, None, logger=_make_logger()).process([event])

        provider.extract.assert_not_called()
        assert [t.text for t in event.tags] == ["trivia", "quiz night"]

    def test_an_ordinary_event_is_still_overwritten(self):
        event = _make_event(summary="stale summary")
        provider = _make_provider(summary="fresh summary")

        ExtractionStage(provider, None, logger=_make_logger()).process([event])

        assert event.summary == "fresh summary"


class TestListingCategoryReachesTheModel:
    """A category recorded but never emitted is worse than one never recorded.

    Dropping the fake `Category: Music` description without emitting the real
    thing left a music listing's input as a bare performer's name — strictly
    less than before, and the prompt's rule about "Event category" referring to
    nothing.
    """

    def test_the_category_is_a_labelled_line_of_its_own(self):
        event = _make_event(
            title="Fred Ellsworth",
            description=None,
            metadata={"listing_category": "Music"},
        )

        assert extraction_input(event) == "Fred Ellsworth\nEvent category: Music"

    def test_an_event_without_one_is_unchanged(self):
        event = _make_event(title="Jazz Night", description="A night of jazz.")

        assert extraction_input(event) == "Jazz Night\nA night of jazz."

    def test_a_real_description_and_a_category_both_appear(self):
        event = _make_event(
            title="Show",
            description="Doors at seven.",
            metadata={"listing_category": "Music"},
        )

        assert extraction_input(event) == "Show\nDoors at seven.\nEvent category: Music"

    def test_the_hash_changes_when_the_category_does(self):
        """Otherwise a corrected category would never trigger a re-extraction."""
        without = _make_event(title="Fred Ellsworth", description=None)
        with_music = _make_event(
            title="Fred Ellsworth", description=None,
            metadata={"listing_category": "Music"},
        )

        assert extraction_input_hash(without) != extraction_input_hash(with_music)
