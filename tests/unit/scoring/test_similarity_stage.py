"""Unit tests for SimilarityStage."""

from __future__ import annotations

from datetime import datetime, timezone
import math

from src.config import ScoringConfig
from src.models.event import Event
from src.models.tag import Tag
from src.scoring.preferences import PreferenceSet, UserPreference
from src.scoring.similarity_stage import SimilarityStage
from src.utils.vectors import encode_vector

_NOW = datetime(2026, 6, 15, tzinfo=timezone.utc)


def _at(s):
    return [s, math.sqrt(max(0.0, 1.0 - s**2))]


def _prefs():
    return PreferenceSet(
        likes=[UserPreference("like", "general", "karaoke", _at(0.95))],
        dislikes=[UserPreference("dislike", "general", "bars", _at(0.30))],
    )


def _event(event_id="e1", tags=None, embedded=True):
    event = Event(
        event_id=event_id, source_event_candidates=[], source_type="apify",
        created_at=_NOW, updated_at=_NOW, tags=tags or [Tag("karaoke")],
    )
    if embedded:
        event.tag_embeddings = [encode_vector(_at(1.0)) for _ in event.tags]
    return event


def _stage(prefs=None, cfg=None):
    return SimilarityStage(preferences=prefs or _prefs(), config=cfg or ScoringConfig())


def test_similarity_attached_to_each_event():
    events = [_event("a"), _event("b")]

    _stage().process(events)

    assert all(e.similarity is not None for e in events)


def test_score_reflects_preferences():
    event = _event()

    _stage().process([event])

    assert event.similarity.base_score > 0
    assert event.similarity.match == "yes"


def test_reasons_attached():
    event = _event()

    _stage().process([event])

    assert [r.tag for r in event.similarity.reasons] == ["karaoke"]


def test_events_returned_for_chaining():
    events = [_event("a")]

    assert _stage().process(events) == events


def test_unembedded_event_scores_zero_without_raising():
    event = _event(embedded=False)

    _stage().process([event])

    assert event.similarity.base_score == 0.0


def test_preferences_are_shared_across_events_not_reloaded():
    """The stage holds one PreferenceSet; scoring must not mutate it."""
    prefs = _prefs()
    stage = SimilarityStage(preferences=prefs, config=ScoringConfig())

    stage.process([_event("a"), _event("b"), _event("c")])

    assert len(prefs.likes) == 1
    assert prefs.likes[0].embedding == _at(0.95)


def test_rescoring_replaces_the_previous_result():
    event = _event()
    _stage().process([event])
    first = event.similarity

    _stage(cfg=ScoringConfig(summary_weight=0.9)).process([event])

    assert event.similarity is not first


def test_empty_batch_is_a_no_op():
    assert _stage().process([]) == []
