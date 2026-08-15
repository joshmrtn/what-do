"""Unit tests for ExtractionStage."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
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
from src.storage.events import validate_tag_vectors
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


def _make_provider(tags=None, summary="A great event.", setting="unknown", degradation=None):

    provider = MagicMock()
    provider.extract.return_value = ExtractionResult(
        title="Extracted Title",
        venue="Extracted Venue",
        start_time=datetime(2026, 6, 22, 20, 0, 0),
        end_time=None,
        # `is None`, not `or` — an empty list is exactly what a degraded reply
        # returns, and `or` would quietly hand it the five-tag default instead.
        tags=[Tag(text=t) for t in
              ["jazz", "live music", "evening", "venue", "weekend"]]
        if tags is None else tags,
        summary=summary,
        model="fake-extraction-model",
        prompt_version="fakever0",
        degradation=degradation,
        setting=setting,
    )
    return provider


def _degraded_provider(reason="tag count 0 is below minimum 1", tags=None, summary=None):
    """A provider whose model answered, and whose answer fell short."""
    return _make_provider(tags=[] if tags is None else tags, summary=summary, degradation=reason)


def _make_fetcher(content: bytes = b"fake image bytes"):
    fetcher = MagicMock()
    fetcher.fetch.return_value = content
    return fetcher


# ---------------------------------------------------------------------------
# Provenance — which model and prompt produced the tags in front of us
# ---------------------------------------------------------------------------


def test_extraction_records_the_model_and_prompt_that_ran():
    provider = _make_provider()
    stage = ExtractionStage(provider, None, _make_logger(), get_now=_now)
    event = _make_event()

    stage.process([event])

    assert event.extraction_model == "fake-extraction-model"
    assert event.extraction_prompt_version == "fakever0"


def test_a_skipped_extraction_leaves_provenance_intact():
    """The path nearly every event on a normal night takes.

    Blanking provenance here would leave the refit with exactly the holes it has
    today, from code that reads as correct — the event keeps its tags, and the
    only thing lost is the record of what produced them.
    """
    provider = _make_provider()
    stage = ExtractionStage(provider, None, _make_logger(), get_now=_now)
    event = _make_event()

    stage.process([event])
    stage.process([event])

    assert provider.extract.call_count == 1, "the second pass re-extracted"
    assert event.extraction_model == "fake-extraction-model"
    assert event.extraction_prompt_version == "fakever0"


def test_a_failed_extraction_records_no_provenance():
    """Nothing produced these tags, so nothing may claim to have."""
    provider = MagicMock()
    provider.extract.side_effect = ExtractionError("model unavailable")
    stage = ExtractionStage(provider, None, _make_logger(), get_now=_now)
    event = _make_event()

    stage.process([event])

    assert event.extraction_model is None
    assert event.extraction_prompt_version is None


def test_a_failed_re_extraction_does_not_erase_earlier_provenance():
    """A retry that fails must not downgrade a row that was fine yesterday."""
    provider = _make_provider()
    stage = ExtractionStage(provider, None, _make_logger(), get_now=_now)
    event = _make_event()
    stage.process([event])

    event.description = "An entirely rewritten description, so the hash differs."
    provider.extract.side_effect = ExtractionError("model unavailable")
    stage.process([event])

    assert event.extraction_model == "fake-extraction-model"
    assert event.extraction_prompt_version == "fakever0"


def test_re_extraction_replaces_provenance_with_what_just_ran():
    """Provenance describes the tags the event currently has, not its history."""
    provider = _make_provider()
    stage = ExtractionStage(provider, None, _make_logger(), get_now=_now)
    event = _make_event()
    stage.process([event])

    event.description = "An entirely rewritten description, so the hash differs."
    provider.extract.return_value = ExtractionResult(
        title=None, venue=None, start_time=None, end_time=None,
        tags=[Tag(text="trivia")], summary="A trivia night.",
        model="gemma4:e2b", prompt_version="newver01", degradation=None,
    )
    stage.process([event])

    assert event.extraction_model == "gemma4:e2b"
    assert event.extraction_prompt_version == "newver01"


def test_synthetic_events_record_no_extraction_provenance():
    """Their tags are authored, so no model or prompt produced them."""
    provider = _make_provider()
    stage = ExtractionStage(provider, None, _make_logger(), get_now=_now)
    event = _make_event(source_type=SYNTHETIC, tags=[Tag(text="board games")])

    stage.process([event])

    assert event.extraction_model is None
    assert event.extraction_prompt_version is None


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


def test_an_unavailable_provider_leaves_the_event_alone_and_continues():
    """No flag is written any more. It was read nowhere, so it marked a failure
    that nothing downstream could see; an absent hash is the real record, and
    it is the one that gets the event retried."""
    provider = MagicMock()
    provider.extract.side_effect = ExtractionError("LLM confused")
    stage = ExtractionStage(provider=provider, image_fetcher=None, logger=_make_logger())

    event = _make_event()
    results = stage.process([event])

    assert results[0].metadata == {}
    assert results[0].tags == []
    assert results[0].extraction_input_hash is None


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
            tags=[Tag(text=c) for c in "abcde"], summary="Good.",
            model="fake-extraction-model", prompt_version="fakever0",
            degradation=None,
        ),
    ]

    event1 = _make_event(event_id="evt-fail")
    event2 = _make_event(event_id="evt-ok")

    stage = ExtractionStage(provider=provider2, image_fetcher=None, logger=_make_logger())
    results = stage.process([event1, event2])

    assert len(results) == 2
    assert results[0].extraction_input_hash is None
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
        model="fake-extraction-model",
        prompt_version="fakever0",
        degradation=None,
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


class TestReExtractionOfAStoredEvent:
    """An event arriving from storage carries the vectors of its stored tags.

    This is the normal production path — almost every event a batch extracts has
    been extracted on a previous night — and no test exercised it. The 2026-08-12
    batch died on it: `min_tags` 5→1 made re-extraction return a different number
    of tags for the first time, and the stale vectors made the event unwritable.
    """

    def _stored(self) -> Event:
        """Five tags and five vectors, as a previous night left them."""
        event = _make_event(extraction_input_hash="stale-hash")
        event.replace_tags([Tag(text=t) for t in ["karaoke", "trivia", "bar", "pub", "evening"]])
        event.attach_tag_embeddings([b"v1", b"v2", b"v3", b"v4", b"v5"])
        return event

    def test_re_extraction_drops_vectors_describing_the_old_tags(self):
        event = self._stored()
        stage = ExtractionStage(
            _make_provider(tags=[Tag(text="trivia")]),
            None,
            _make_logger(),
            get_now=_now,
        )

        stage.process([event])

        assert event.tags == [Tag(text="trivia")]
        assert event.tag_embeddings == []

    def test_the_re_extracted_event_can_be_persisted(self):
        """The checkpoint that refused 64 events must now succeed.

        Extraction costs minutes an event; embedding costs about a second a tag.
        Refusing the write threw away the expensive half to protect the cheap
        one, so this asserts the expensive half survives.
        """
        event = self._stored()
        saved: list[Event] = []
        stage = ExtractionStage(
            _make_provider(tags=[Tag(text="trivia")]),
            None,
            _make_logger(),
            get_now=_now,
            save_fn=saved.append,
        )

        stage.process([event])

        assert len(saved) == 1
        validate_tag_vectors(saved[0])  # must not raise

    def test_a_skipped_event_keeps_the_vectors_it_arrived_with(self):
        """Nothing is dropped when extraction does not run.

        The skip rule is what keeps a nightly batch under an hour; re-embedding
        every stored event because it was *considered* would undo that.
        """
        event = self._stored()
        event.extraction_input_hash = extraction_input_hash(event)
        stage = ExtractionStage(_make_provider(), None, _make_logger(), get_now=_now)

        stage.process([event])

        assert event.tag_embeddings == [b"v1", b"v2", b"v3", b"v4", b"v5"]
        assert len(event.tags) == 5

    def test_a_failed_extraction_keeps_the_vectors_it_arrived_with(self):
        """A model that errored has replaced nothing, so nothing is stale."""
        event = self._stored()
        provider = MagicMock()
        provider.extract.side_effect = ExtractionError("model unavailable")
        stage = ExtractionStage(provider, None, _make_logger(), get_now=_now)

        stage.process([event])

        assert event.tag_embeddings == [b"v1", b"v2", b"v3", b"v4", b"v5"]

    def test_re_extraction_drops_the_stale_summary_vector(self):
        event = self._stored()
        event.summary = "An evening of karaoke and trivia."
        event.summary_embedding = b"v-summary"
        stage = ExtractionStage(
            _make_provider(tags=[Tag(text="trivia")], summary="A trivia night."),
            None,
            _make_logger(),
            get_now=_now,
        )

        stage.process([event])

        assert event.summary == "A trivia night."
        assert event.summary_embedding is None

    def test_an_authored_summary_keeps_its_vector(self):
        """The model never replaced it, so there is nothing stale to drop."""
        event = self._stored()
        event.summary = "Trivia at The Paddle Inn."
        event.summary_embedding = b"v-authored"
        event.metadata["authored_summary"] = True
        stage = ExtractionStage(
            _make_provider(tags=[Tag(text="trivia")], summary="Invented prose."),
            None,
            _make_logger(),
            get_now=_now,
        )

        stage.process([event])

        assert event.summary == "Trivia at The Paddle Inn."
        assert event.summary_embedding == b"v-authored"


class TestADegradedExtractionIsRecordedAndFinished:
    """A model that answered thinly has finished with this input.

    Measured: the same eight events failed on 2026-08-12 and again on 08-13,
    each burning minutes of model time to produce an identical rejection. The
    shortfall is deterministic, so the run is done — the hash goes down and the
    event stops being re-paid for nightly. That is the opposite of a transport
    failure, which may well succeed tomorrow and must stay retryable.
    """

    def test_a_degraded_extraction_records_its_provenance(self):
        stage = ExtractionStage(_degraded_provider(), None, _make_logger(), get_now=_now)
        event = _make_event()

        stage.process([event])

        assert event.extraction_model == "fake-extraction-model"
        assert event.extraction_prompt_version == "fakever0"

    def test_a_degraded_extraction_writes_the_input_hash(self):
        stage = ExtractionStage(_degraded_provider(), None, _make_logger(), get_now=_now)
        event = _make_event()

        stage.process([event])

        assert event.extraction_input_hash == extraction_input_hash(event)

    def test_a_degraded_extraction_is_not_re_paid_on_the_next_pass(self):
        provider = _degraded_provider()
        stage = ExtractionStage(provider, None, _make_logger(), get_now=_now)
        event = _make_event()

        stage.process([event])
        stage.process([event])

        assert provider.extract.call_count == 1

    def test_a_degraded_extraction_records_what_it_fell_short_on(self):
        stage = ExtractionStage(
            _degraded_provider(reason="tag count 0 is below minimum 1; summary field is missing"),
            None,
            _make_logger(),
            get_now=_now,
        )
        event = _make_event()

        stage.process([event])

        assert event.extraction_degradation == (
            "tag count 0 is below minimum 1; summary field is missing"
        )

    def test_a_clean_extraction_records_no_degradation(self):
        stage = ExtractionStage(_make_provider(), None, _make_logger(), get_now=_now)
        event = _make_event()

        stage.process([event])

        assert event.extraction_degradation is None

    def test_a_later_clean_extraction_clears_an_earlier_degradation(self):
        """The field describes the tags the event has now, like its provenance."""
        provider = _degraded_provider()
        stage = ExtractionStage(provider, None, _make_logger(), get_now=_now)
        event = _make_event()
        stage.process([event])

        event.description = "An entirely rewritten description, so the hash differs."
        provider.extract.return_value = _make_provider().extract.return_value
        stage.process([event])

        assert event.extraction_degradation is None

    def test_it_wipes_tags_it_can_no_longer_justify(self):
        """The seven events this was measured on kept tags from the forced-five
        prompt — `genre`, `artist`, a title echo — and ranked on them while
        claiming tonight's provenance. Zero tags is the honest position, and
        `tag_confidence` puts zero evidence at mid-ranking, where it belongs."""
        stage = ExtractionStage(_degraded_provider(), None, _make_logger(), get_now=_now)
        event = _make_event(tags=[Tag(text="genre", weight=0.5), Tag(text="artist", weight=0.3)])

        stage.process([event])

        assert event.tags == []

    def test_the_usable_half_of_a_degraded_reply_is_still_adopted(self):
        stage = ExtractionStage(
            _degraded_provider(summary="This is a stand-up comedy event."),
            None,
            _make_logger(),
            get_now=_now,
        )
        event = _make_event()

        stage.process([event])

        assert event.summary == "This is a stand-up comedy event."

    def test_a_transport_failure_still_leaves_the_event_retryable(self):
        """The distinction the whole design turns on: no hash, so it runs again."""
        provider = MagicMock()
        provider.extract.side_effect = ExtractionError("model unavailable")
        stage = ExtractionStage(provider, None, _make_logger(), get_now=_now)
        event = _make_event()

        stage.process([event])

        assert event.extraction_input_hash is None
        assert event.extraction_degradation is None

    def test_the_dead_failure_flag_is_gone(self):
        """Asserted on the transport path, which is the one that wrote it — a
        degraded event never carried the flag, so checking there would pass
        against code that still sets it. It was read nowhere in `src/`, so
        every failure it marked was invisible to everything downstream."""
        provider = MagicMock()
        provider.extract.side_effect = ExtractionError("model unavailable")
        stage = ExtractionStage(provider, None, _make_logger(), get_now=_now)
        event = _make_event()

        stage.process([event])

        assert "llm_extraction_failed" not in event.metadata


class TestTheExtractionBudget:
    """Extraction is the only stage measured in minutes an event, so it is the
    only one that needs bounding.

    A cold start against a 45-day horizon was measured at 14.9h and 19.7h, and
    changing `extraction_input` re-extracts the whole corpus — ~1,440 events,
    roughly 45h. Without a bound those land in one night. With one they land
    over several, soonest-first, so tonight is always ready tomorrow.
    """

    @staticmethod
    def _clock(minutes_per_event):
        """A clock that advances only when the provider is called."""
        state = {"now": datetime(2026, 6, 22, 2, 0, 0, tzinfo=timezone.utc)}

        def get_now():
            return state["now"]

        def tick(*_args, **_kwargs):
            state["now"] += timedelta(minutes=minutes_per_event)
            return _make_provider().extract.return_value

        return get_now, tick

    def test_it_stops_once_the_budget_is_spent(self):
        """Three, not two. Events start at 0, 4 and 8 minutes and the budget is
        10, so the third starts inside it and finishes at 12 — the bounded
        overshoot the check-before rule buys deliberately."""
        get_now, tick = self._clock(minutes_per_event=4)
        provider = MagicMock()
        provider.extract.side_effect = tick
        stage = ExtractionStage(
            provider, None, _make_logger(), get_now=get_now, budget_minutes=10
        )
        events = [_make_event(event_id=f"evt-{i}") for i in range(5)]

        stage.process(events)

        assert provider.extract.call_count == 3

    def test_a_deferred_event_stays_retryable(self):
        """No hash, so the next run picks it up — the same mechanism an
        unavailable provider already uses."""
        get_now, tick = self._clock(minutes_per_event=4)
        provider = MagicMock()
        provider.extract.side_effect = tick
        stage = ExtractionStage(
            provider, None, _make_logger(), get_now=get_now, budget_minutes=10
        )
        events = [_make_event(event_id=f"evt-{i}") for i in range(5)]

        stage.process(events)

        assert events[-1].extraction_input_hash is None
        assert events[-1].extraction_model is None

    def test_no_budget_extracts_everything(self):
        """Today's behaviour, and what almost every caller wants."""
        provider = _make_provider()
        stage = ExtractionStage(provider, None, _make_logger(), get_now=_now)
        events = [_make_event(event_id=f"evt-{i}") for i in range(5)]

        stage.process(events)

        assert provider.extract.call_count == 5

    def test_the_budget_is_checked_before_an_event_not_during_it(self):
        """One event may overshoot; the alternative is abandoning model time
        already spent, which is the thing the checkpoint exists to protect."""
        get_now, tick = self._clock(minutes_per_event=30)
        provider = MagicMock()
        provider.extract.side_effect = tick
        stage = ExtractionStage(
            provider, None, _make_logger(), get_now=get_now, budget_minutes=10
        )
        events = [_make_event(event_id=f"evt-{i}") for i in range(3)]

        stage.process(events)

        assert provider.extract.call_count == 1
        assert events[0].extraction_input_hash is not None

    def test_a_skipped_event_is_never_counted_as_deferred(self):
        """An event whose hash matches never wanted the model, so a spent budget
        has not denied it anything. Counting it would inflate the one number
        that says whether the budget is set too low.

        Asserted on the log rather than the call count, because a skip consumes
        no model time either way — the call count cannot tell the orderings
        apart, and a test that cannot fail is worse than no test.
        """
        stream = io.StringIO()
        logger = get_logger("test_budget", stream=stream)
        get_now, tick = self._clock(minutes_per_event=4)
        provider = MagicMock()
        provider.extract.side_effect = tick
        # Zero budget: every event that wants the model is deferred immediately.
        stage = ExtractionStage(provider, None, logger, get_now=get_now, budget_minutes=0)
        done = _make_event(event_id="already-done", tags=[Tag(text="jazz")])
        done.extraction_input_hash = extraction_input_hash(done)

        stage.process([done, _make_event(event_id="evt-0")])

        provider.extract.assert_not_called()
        assert "1 deferred" in stream.getvalue()


class TestExtractionOrder:
    """Soonest first, so tonight is ready tomorrow however deep the backlog.

    It also gives the unavailable-provider path the right failure mode for
    free: whatever a run does not reach is the stuff furthest out.
    """

    @staticmethod
    def _order_recording_provider():
        """Records the title of each event as the model is asked about it."""
        seen: list[str] = []
        provider = MagicMock()

        def record(text, **_kwargs):
            seen.append(text.split("\n")[0])
            return _make_provider().extract.return_value

        provider.extract.side_effect = record
        return provider, seen

    def test_events_are_extracted_soonest_first(self):
        provider, seen = self._order_recording_provider()
        stage = ExtractionStage(provider, None, _make_logger(), get_now=_now)
        events = [
            _make_event(event_id="c", title="Third", start_time=datetime(2026, 7, 3, tzinfo=timezone.utc)),
            _make_event(event_id="a", title="First", start_time=datetime(2026, 7, 1, tzinfo=timezone.utc)),
            _make_event(event_id="b", title="Second", start_time=datetime(2026, 7, 2, tzinfo=timezone.utc)),
        ]

        stage.process(events)

        assert seen == ["First", "Second", "Third"]

    def test_the_returned_list_keeps_its_original_order(self):
        """`process` hands this list straight back to the orchestrator, which
        saves it and passes it on. Re-ordering the caller's list is a side
        effect nothing asked for."""
        provider, _ = self._order_recording_provider()
        stage = ExtractionStage(provider, None, _make_logger(), get_now=_now)
        events = [
            _make_event(event_id="c", start_time=datetime(2026, 7, 3, tzinfo=timezone.utc)),
            _make_event(event_id="a", start_time=datetime(2026, 7, 1, tzinfo=timezone.utc)),
        ]

        returned = stage.process(events)

        assert [e.event_id for e in returned] == ["c", "a"]

    def test_undated_events_go_last(self):
        provider, seen = self._order_recording_provider()
        stage = ExtractionStage(provider, None, _make_logger(), get_now=_now)
        events = [
            _make_event(event_id="u", title="Undated", start_time=None),
            _make_event(event_id="a", title="Dated", start_time=datetime(2026, 7, 1, tzinfo=timezone.utc)),
        ]

        stage.process(events)

        assert seen == ["Dated", "Undated"]

    def test_a_spent_budget_defers_the_furthest_out(self):
        get_now, tick = TestTheExtractionBudget._clock(minutes_per_event=4)
        provider = MagicMock()
        provider.extract.side_effect = tick
        # Three, not five: starts happen at 0 and 4 minutes, so a budget of 5
        # would admit both and prove nothing about the order.
        stage = ExtractionStage(
            provider, None, _make_logger(), get_now=get_now, budget_minutes=3
        )
        soon = _make_event(event_id="soon", start_time=datetime(2026, 7, 1, tzinfo=timezone.utc))
        later = _make_event(event_id="later", start_time=datetime(2026, 9, 1, tzinfo=timezone.utc))

        stage.process([later, soon])

        assert soon.extraction_input_hash is not None
        assert later.extraction_input_hash is None

    def test_mixed_naive_and_aware_start_times_do_not_raise(self):
        """Sources genuinely differ — JSON-LD states an offset, HTML listings do
        not — and comparing the two forms directly raises."""
        provider, seen = self._order_recording_provider()
        stage = ExtractionStage(provider, None, _make_logger(), get_now=_now)
        events = [
            _make_event(event_id="naive", title="Naive", start_time=datetime(2026, 7, 5)),
            _make_event(event_id="aware", title="Aware", start_time=datetime(2026, 7, 1, tzinfo=timezone.utc)),
        ]

        stage.process(events)

        assert len(seen) == 2


def test_the_stage_reports_how_many_events_it_deferred():
    """The orchestrator cannot count these itself: an event with no hash may
    have been deferred, or the provider may have been unavailable, and those
    want different responses. A deferral count that stays high run after run is
    the signal the budget is set too low."""
    get_now, tick = TestTheExtractionBudget._clock(minutes_per_event=4)
    provider = MagicMock()
    provider.extract.side_effect = tick
    stage = ExtractionStage(provider, None, _make_logger(), get_now=get_now, budget_minutes=3)
    events = [_make_event(event_id=f"evt-{i}") for i in range(4)]

    stage.process(events)

    assert stage.deferred == 3


def test_the_deferral_count_resets_between_runs():
    """It describes the run just finished, not every run since startup."""
    provider = _make_provider()
    stage = ExtractionStage(provider, None, _make_logger(), get_now=_now, budget_minutes=0)
    stage.process([_make_event(event_id="evt-0")])
    assert stage.deferred == 1

    stage_again = ExtractionStage(provider, None, _make_logger(), get_now=_now)
    stage_again.process([_make_event(event_id="evt-1")])
    assert stage_again.deferred == 0


def test_extraction_does_not_re_extract_for_a_change_it_made_itself():
    """The venue the model fills is part of `extraction_input`, so a hash taken
    before the fills describes text the event no longer has — and the next run
    re-extracts every event whose venue we completed, at minutes apiece, for a
    change we caused. Surfaced the moment the venue entered the input."""
    provider = _make_provider()  # its result carries venue="Extracted Venue"
    stage = ExtractionStage(provider, None, _make_logger(), get_now=_now)
    event = _make_event(venue=None)

    stage.process([event])
    assert event.venue == "Extracted Venue", "the fill this test is about did not happen"
    stage.process([event])

    assert provider.extract.call_count == 1


class TestTheCategoryFallback:
    """When extraction yields no usable tag, the listing's own heading is the
    most honest tag available.

    Safe by construction: `_CARRIED_CATEGORIES` already withholds `Other` and
    `Karaoke & trivia`, so a heading only reaches `listing_category` when it
    says something. `music` for a bare name under Music beats both invention
    and silence.
    """

    def test_a_tagless_result_falls_back_to_the_listing_category(self):
        stage = ExtractionStage(_degraded_provider(), None, _make_logger(), get_now=_now)
        event = _make_event(metadata={"listing_category": "Music"})

        stage.process([event])

        assert [t.text for t in event.tags] == ["music"]

    def test_the_fallback_tag_carries_full_weight(self):
        """Weight is centrality, not confidence. It is the only thing we know
        about the event, so it is entirely central; `tag_confidence` is the
        field that says how little we know, and it will already be low."""
        stage = ExtractionStage(_degraded_provider(), None, _make_logger(), get_now=_now)
        event = _make_event(metadata={"listing_category": "Music"})

        stage.process([event])

        assert event.tags[0].weight == 1.0

    def test_the_fallback_is_recorded_in_the_degradation(self):
        """The refit fits against *model* behaviour, and this tag is not
        something the model produced — counting it would teach the curve that
        the model emits tags it never emitted."""
        stage = ExtractionStage(_degraded_provider(), None, _make_logger(), get_now=_now)
        event = _make_event(metadata={"listing_category": "Music"})

        stage.process([event])

        assert "tag count 0" in event.extraction_degradation
        assert "listing category" in event.extraction_degradation

    def test_no_category_means_no_fallback(self):
        """`Other` and `Karaoke & trivia` never reach the field, so an event
        with nothing here had a heading worth withholding — or none at all."""
        stage = ExtractionStage(_degraded_provider(), None, _make_logger(), get_now=_now)
        event = _make_event()

        stage.process([event])

        assert event.tags == []
        assert "listing category" not in (event.extraction_degradation or "")

    def test_a_result_with_tags_is_left_alone(self):
        stage = ExtractionStage(_make_provider(), None, _make_logger(), get_now=_now)
        event = _make_event(metadata={"listing_category": "Music"})

        stage.process([event])

        assert "music" not in [t.text for t in event.tags]
        assert event.extraction_degradation is None

    def test_a_degraded_result_that_still_has_a_tag_is_left_alone(self):
        """The shortfall may be the summary. One real tag is not nothing, and
        appending the category would dilute it."""
        provider = _degraded_provider(
            reason="summary field is missing or not a string",
            tags=[Tag(text="comedy", weight=1.0)],
        )
        stage = ExtractionStage(provider, None, _make_logger(), get_now=_now)
        event = _make_event(metadata={"listing_category": "Music"})

        stage.process([event])

        assert [t.text for t in event.tags] == ["comedy"]


class TestOnlyEventsRankingCanUseAreExtracted:
    """The queue is soonest-first, so an event that has already happened sorts
    to the *front* of it.

    Harmless until 3d made model time scarce. Measured on 2026-08-14: the whole
    480-minute budget went on 273 events that were already over, and the run
    reached nothing rankable at all. The night before did the same. These are
    not stale listings — they were future when ingested and aged into the past
    while waiting in the queue behind a 1,200-event backlog.

    Skipping is safe because extraction only ever *fills* a null `start_time`
    and never corrects one, so a past event's date is the same whether the
    model runs or not; and an event whose date is genuinely not knowable yet is
    undated, which the ranking predicate keeps on discovery age.
    """

    @staticmethod
    def _rankable(event):
        """Stands in for the scheduler's ranking scope.

        The real predicate is `_scope_filter`, which has its own tests and is
        wired in `tests/unit/test_scheduler.py`. This stage's contract is only
        that it asks and obeys.
        """
        return event.start_time is None or event.start_time >= _now()

    def _past(self, **kwargs):
        return _make_event(start_time=_now() - timedelta(days=2), **kwargs)

    def _future(self, **kwargs):
        return _make_event(start_time=_now() + timedelta(days=2), **kwargs)

    def test_an_event_ranking_would_discard_is_never_sent_to_the_model(self):
        provider = _make_provider()
        stage = ExtractionStage(provider, None, _make_logger(), get_now=_now)
        stage.set_scope_fn(self._rankable)

        stage.process([self._past()])

        provider.extract.assert_not_called()

    def test_an_out_of_scope_event_is_skipped_not_dropped(self):
        """This stage returns the list it was given. The orchestrator hands the
        same list to embedding and to `events_repo.replace(stale_ids, events)`,
        so dropping an event here would stop it being persisted at all."""
        stage = ExtractionStage(_make_provider(), None, _make_logger(), get_now=_now)
        stage.set_scope_fn(self._rankable)
        past, future = self._past(event_id="past"), self._future(event_id="future")

        returned = stage.process([past, future])

        assert [e.event_id for e in returned] == ["past", "future"]

    def test_an_event_still_to_come_is_extracted(self):
        provider = _make_provider()
        stage = ExtractionStage(provider, None, _make_logger(), get_now=_now)
        stage.set_scope_fn(self._rankable)

        stage.process([self._future()])

        assert provider.extract.call_count == 1

    def test_an_out_of_scope_event_costs_no_budget_and_is_not_deferred(self):
        """The whole point. A queue full of expired events must not report
        itself as deferred work, or the count that says "the budget is too
        small" says it for events no budget should ever buy."""
        get_now, tick = TestTheExtractionBudget._clock(minutes_per_event=4)
        provider = MagicMock()
        provider.extract.side_effect = tick
        stage = ExtractionStage(
            provider, None, _make_logger(), get_now=get_now, budget_minutes=10
        )
        # The clock this budget runs on, not the module-level one.
        stage.set_scope_fn(lambda e: e.start_time is None or e.start_time >= get_now())
        start = get_now()
        events = [
            _make_event(event_id=f"past-{i}", start_time=start - timedelta(days=2))
            for i in range(5)
        ] + [
            _make_event(event_id=f"soon-{i}", start_time=start + timedelta(days=1))
            for i in range(3)
        ]

        stage.process(events)

        assert provider.extract.call_count == 3, "the budget bought only rankable events"
        assert stage.deferred == 0

    def test_with_no_scope_fn_every_event_is_extracted(self):
        """The default is permissive, and covered rather than assumed."""
        provider = _make_provider()
        stage = ExtractionStage(provider, None, _make_logger(), get_now=_now)

        stage.process([self._past(), self._future()])

        assert provider.extract.call_count == 2

    def test_the_stage_reports_how_many_it_skipped_as_out_of_scope(self):
        """Without this the fix is invisible: a run that silently skips 124
        events looks exactly like one with a smaller backlog."""
        stage = ExtractionStage(_make_provider(), None, _make_logger(), get_now=_now)
        stage.set_scope_fn(self._rankable)

        stage.process([self._past(event_id="a"), self._past(event_id="b"), self._future()])

        assert stage.out_of_scope == 2

    def test_an_already_extracted_past_event_is_not_counted_out_of_scope(self):
        """The scope check sits *after* the hash check, so the count means
        "stale and out of scope" — the work actually saved. Before it, the same
        skip would count every past event ever extracted: ~470 against a real
        saving of 124, a number describing nothing."""
        stage = ExtractionStage(_make_provider(), None, _make_logger(), get_now=_now)
        stage.set_scope_fn(self._rankable)
        done = self._past(event_id="done", tags=[Tag(text="jazz")])
        done.extraction_input_hash = extraction_input_hash(done)

        stage.process([done, self._past(event_id="stale")])

        assert stage.out_of_scope == 1

    def test_the_out_of_scope_count_resets_between_runs(self):
        """It describes the run just finished, as `deferred` does."""
        stage = ExtractionStage(_make_provider(), None, _make_logger(), get_now=_now)
        stage.set_scope_fn(self._rankable)
        stage.process([self._past()])
        assert stage.out_of_scope == 1

        stage.process([self._future()])

        assert stage.out_of_scope == 0


def test_a_venue_the_model_supplies_is_canonicalised():
    """Extraction fills `event.venue` when the listing had none, and that value
    used to reach storage without passing through `_normalize_venue`.

    Measured 2026-08-14: it is why one Seven Gables event was stored as "The
    House of the Seven Gables" while the other fifty-one were "The House Of The
    Seven Gables". The candidates table held a single spelling — the second was
    written by the model, in its own casing, straight onto the event. Two forms
    of one venue then failed `venues_match` and never deduplicated.
    """
    event = _make_event(venue=None)
    provider = _make_provider()
    provider.extract.return_value.venue = "the house of the seven gables"
    stage = ExtractionStage(provider=provider, image_fetcher=None, logger=_make_logger())

    results = stage.process([event])

    assert results[0].venue == "The House Of The Seven Gables"


def test_extraction_records_what_the_model_was_asked():
    """The observation, not just its digest.

    The hash alone made the corpus reconstructable-in-principle and wrong in
    practice: rebuilding the input needs today's builder and today's fields, and
    extraction itself changes the fields. Recorded after the fills, on the same
    terms and for the same reason as the hash.
    """
    event = _make_event(venue=None)
    provider = _make_provider()
    stage = ExtractionStage(provider=provider, image_fetcher=None, logger=_make_logger())

    results = stage.process([event])

    assert results[0].extraction_input is not None
    assert results[0].extraction_input_chars == len(results[0].extraction_input)
    # The venue the model supplied is part of what a later run would be asked,
    # so it has to be inside the recorded text — the hash is taken here too.
    assert "Extracted Venue" in results[0].extraction_input


def test_a_skipped_extraction_does_not_blank_the_recorded_input():
    """Almost every event skips on a normal night. Overwriting the observation
    with None on the skip path would empty the corpus one quiet night."""
    event = _make_event()
    provider = _make_provider()
    stage = ExtractionStage(provider=provider, image_fetcher=None, logger=_make_logger())
    stage.process([event])
    recorded = event.extraction_input

    stage.process([event])  # second run: hash unchanged, so extraction skips

    assert event.extraction_input == recorded
