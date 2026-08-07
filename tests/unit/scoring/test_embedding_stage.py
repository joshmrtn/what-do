"""Unit tests for EmbeddingStage."""

from __future__ import annotations

from datetime import datetime, timezone
import io

import pytest

from src.models.event import Event
from src.models.tag import Tag
from src.scoring.embedding_stage import EmbeddingStage
from src.scoring.embeddings import EmbeddingError
from src.utils.logging import get_logger
from src.utils.vectors import decode_vector


class _FakeProvider:
    """Returns a deterministic vector per text and records every call."""

    def __init__(self, fail_on: str | None = None, dim: int = 4):
        self.calls: list[str] = []
        self._fail_on = fail_on
        self._dim = dim

    def embed(self, text: str) -> list[float]:

        self.calls.append(text)
        if text == self._fail_on:
            raise EmbeddingError("model unavailable")
        seed = float(sum(ord(c) for c in text) % 97)
        return [seed + i for i in range(self._dim)]


def _now() -> datetime:
    return datetime(2026, 6, 15, 12, 0, tzinfo=timezone.utc)


def _event(event_id="e1", tags=None, summary="A live music night.") -> Event:
    return Event(
        event_id=event_id,
        source_event_candidates=[],
        source_type="apify",
        created_at=_now(),
        updated_at=_now(),
        tags=tags if tags is not None else [Tag(text="karaoke"), Tag(text="bar", weight=0.2)],
        summary=summary,
    )


def _stage(provider, stream=None):

    return EmbeddingStage(
        provider=provider,
        logger=get_logger("test", stream=stream or io.StringIO()),
    )


# ---------------------------------------------------------------------------
# Core behaviour
# ---------------------------------------------------------------------------


def test_one_embedding_per_tag_in_tag_order():
    provider = _FakeProvider()
    event = _event(tags=[Tag(text="karaoke"), Tag(text="punk"), Tag(text="bar")])

    _stage(provider).process([event])

    assert provider.calls[:3] == ["karaoke", "punk", "bar"]
    assert len(event.tag_embeddings) == 3


def test_summary_embedded():
    provider = _FakeProvider()
    event = _event(summary="A karaoke night.")

    _stage(provider).process([event])

    assert "A karaoke night." in provider.calls
    assert event.summary_embedding is not None


def test_embeddings_stored_as_bytes():
    event = _event()

    _stage(_FakeProvider()).process([event])

    assert all(isinstance(b, bytes) for b in event.tag_embeddings)
    assert isinstance(event.summary_embedding, bytes)


def test_stored_vector_decodes_to_provider_output():
    provider = _FakeProvider()
    event = _event(tags=[Tag(text="karaoke")], summary=None)

    _stage(provider).process([event])

    expected = _FakeProvider().embed("karaoke")
    assert decode_vector(event.tag_embeddings[0]) == pytest.approx(expected)


def test_events_returned_for_chaining():
    events = [_event("a"), _event("b")]

    result = _stage(_FakeProvider()).process(events)

    assert result == events


# ---------------------------------------------------------------------------
# Memoisation
# ---------------------------------------------------------------------------


def test_repeated_tag_across_events_embedded_once():
    """Tags like 'live music' recur constantly; embedding each is wasted time."""
    provider = _FakeProvider()
    events = [
        _event("a", tags=[Tag(text="live music")], summary=None),
        _event("b", tags=[Tag(text="live music")], summary=None),
    ]

    _stage(provider).process(events)

    assert provider.calls == ["live music"]


def test_memoised_events_still_each_get_their_vector():
    provider = _FakeProvider()
    events = [
        _event("a", tags=[Tag(text="live music")], summary=None),
        _event("b", tags=[Tag(text="live music")], summary=None),
    ]

    _stage(provider).process(events)

    assert events[0].tag_embeddings[0] == events[1].tag_embeddings[0]
    assert len(events[1].tag_embeddings) == 1


def test_duplicate_tag_within_one_event_embedded_once():
    provider = _FakeProvider()
    event = _event(tags=[Tag(text="karaoke"), Tag(text="karaoke")], summary=None)

    _stage(provider).process([event])

    assert provider.calls == ["karaoke"]
    assert len(event.tag_embeddings) == 2


# ---------------------------------------------------------------------------
# Idempotence and skipping
# ---------------------------------------------------------------------------


def test_already_embedded_event_is_skipped():
    provider = _FakeProvider()
    event = _event(tags=[Tag(text="karaoke")], summary=None)
    _stage(provider).process([event])
    provider.calls.clear()

    _stage(provider).process([event])

    assert provider.calls == []


def test_event_without_tags_is_flagged_and_not_embedded():
    provider = _FakeProvider()
    event = _event(tags=[], summary=None)

    _stage(provider).process([event])

    assert provider.calls == []
    assert event.metadata["embedding_skipped"] is True


def test_event_without_summary_still_embeds_tags():
    provider = _FakeProvider()
    event = _event(tags=[Tag(text="karaoke")], summary=None)

    _stage(provider).process([event])

    assert len(event.tag_embeddings) == 1
    assert event.summary_embedding is None


def test_blank_summary_treated_as_absent():
    provider = _FakeProvider()
    event = _event(tags=[Tag(text="karaoke")], summary="   ")

    _stage(provider).process([event])

    assert event.summary_embedding is None
    assert provider.calls == ["karaoke"]


# ---------------------------------------------------------------------------
# Failure handling — unlike preferences, one bad event must not stop the batch
# ---------------------------------------------------------------------------


def test_failed_event_is_flagged_and_batch_continues():
    provider = _FakeProvider(fail_on="cursed")
    events = [
        _event("a", tags=[Tag(text="cursed")], summary=None),
        _event("b", tags=[Tag(text="karaoke")], summary=None),
    ]

    _stage(provider).process(events)

    assert events[0].metadata["embedding_failed"] is True
    assert events[0].tag_embeddings == []
    assert len(events[1].tag_embeddings) == 1


def test_failure_is_logged_with_event_id():
    provider = _FakeProvider(fail_on="cursed")
    stream = io.StringIO()
    event = _event("evt-42", tags=[Tag(text="cursed")], summary=None)

    _stage(provider, stream).process([event])

    assert "evt-42" in stream.getvalue()


def test_partial_tag_failure_leaves_no_half_embedded_event():
    """A half-embedded event would score against a subset of its own tags."""
    provider = _FakeProvider(fail_on="cursed")
    event = _event(tags=[Tag(text="karaoke"), Tag(text="cursed")], summary=None)

    _stage(provider).process([event])

    assert event.metadata["embedding_failed"] is True
    assert event.tag_embeddings == []


def test_summary_failure_does_not_discard_tag_embeddings():
    provider = _FakeProvider(fail_on="cursed summary")
    event = _event(tags=[Tag(text="karaoke")], summary="cursed summary")

    _stage(provider).process([event])

    assert len(event.tag_embeddings) == 1
    assert event.summary_embedding is None
    assert event.metadata["embedding_failed"] is True


# ---------------------------------------------------------------------------
# Input-hash skip rule
# ---------------------------------------------------------------------------


def test_re_extracted_tags_are_re_embedded():
    """Vectors describing tags the event no longer has would silently misrank it."""
    provider = _FakeProvider()
    stage = EmbeddingStage(provider, get_logger("t", stream=io.StringIO()))
    event = _event(tags=[Tag(text="karaoke", weight=1.0)], summary="Karaoke night.")

    stage.process([event])
    first = len(provider.calls)

    event.tags = [Tag(text="punk", weight=1.0)]
    event.summary = "A punk show."
    stage.process([event])

    assert len(provider.calls) > first
    assert "punk" in provider.calls
    assert len(event.tag_embeddings) == 1


def test_unchanged_tags_are_not_re_embedded():
    provider = _FakeProvider()
    stage = EmbeddingStage(provider, get_logger("t", stream=io.StringIO()))
    event = _event(tags=[Tag(text="karaoke", weight=1.0)], summary="Karaoke night.")

    stage.process([event])
    first = len(provider.calls)
    stage.process([event])

    assert len(provider.calls) == first
