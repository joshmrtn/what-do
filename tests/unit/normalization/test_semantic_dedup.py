"""Unit tests for semantic deduplication (Pass 2, post-embedding)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import math

from src.config import DeduplicationConfig
from src.models.event import Event
from src.models.tag import Tag
from src.normalization.semantic_dedup import SemanticDeduplicationEngine
from src.utils.vectors import encode_vector

_BASE = datetime(2026, 6, 15, 20, 0, tzinfo=timezone.utc)


def _vec_at(similarity: float) -> bytes:
    """Encode a unit vector whose cosine with [1, 0] is exactly `similarity`."""
    return encode_vector([similarity, math.sqrt(max(0.0, 1.0 - similarity**2))])


def _reference() -> bytes:
    return encode_vector([1.0, 0.0])


def _event(
    event_id="e1",
    *,
    summary_embedding=None,
    venue="The Vault",
    start_time=_BASE,
    candidates=None,
    title="Jazz Night",
    summary="An evening of jazz.",
    **kwargs,
) -> Event:
    return Event(
        event_id=event_id,
        source_event_candidates=candidates if candidates is not None else [event_id],
        source_type="apify",
        created_at=_BASE,
        updated_at=_BASE,
        title=title,
        venue=venue,
        start_time=start_time,
        summary=summary,
        summary_embedding=summary_embedding,
        **kwargs,
    )


def _cfg(threshold=0.92, window=2.0) -> DeduplicationConfig:
    return DeduplicationConfig(semantic_threshold=threshold, time_window_hours=window)


def _dedupe(events, cfg=None):
    return SemanticDeduplicationEngine().deduplicate(events, cfg or _cfg())


# ---------------------------------------------------------------------------
# Core merging
# ---------------------------------------------------------------------------


def test_same_meaning_different_wording_merged():
    """The whole point of Pass 2: fuzzy title matching missed these."""
    events = [
        _event("a", title="Jazz Night", summary_embedding=_reference()),
        _event("b", title="An Evening of Jazz", summary_embedding=_vec_at(0.97)),
    ]

    assert len(_dedupe(events)) == 1


def test_below_threshold_not_merged():
    events = [
        _event("a", summary_embedding=_reference()),
        _event("b", summary_embedding=_vec_at(0.80)),
    ]

    assert len(_dedupe(events)) == 2


def test_threshold_read_from_config():
    events = [
        _event("a", summary_embedding=_reference()),
        _event("b", summary_embedding=_vec_at(0.90)),
    ]

    assert len(_dedupe(events, _cfg(threshold=0.85))) == 1
    assert len(_dedupe(events, _cfg(threshold=0.95))) == 2


def test_three_similar_events_collapse_to_one():
    events = [
        _event("a", summary_embedding=_reference()),
        _event("b", summary_embedding=_vec_at(0.98)),
        _event("c", summary_embedding=_vec_at(0.96)),
    ]

    result = _dedupe(events)

    assert len(result) == 1
    assert sorted(result[0].source_event_candidates) == ["a", "b", "c"]


# ---------------------------------------------------------------------------
# Structural guards — semantic similarity alone would merge recurrences
# ---------------------------------------------------------------------------


def test_identical_summaries_at_different_venues_not_merged():
    events = [
        _event("a", venue="The Vault", summary_embedding=_reference()),
        _event("b", venue="O'Neill's", summary_embedding=_reference()),
    ]

    assert len(_dedupe(events)) == 2


def test_recurring_event_on_different_nights_not_merged():
    """Weekly karaoke has near-identical summaries every week."""
    events = [
        _event("a", start_time=_BASE, summary_embedding=_reference()),
        _event("b", start_time=_BASE + timedelta(days=7), summary_embedding=_reference()),
    ]

    assert len(_dedupe(events)) == 2


def test_same_night_within_time_window_merged():
    events = [
        _event("a", start_time=_BASE, summary_embedding=_reference()),
        _event("b", start_time=_BASE + timedelta(hours=1), summary_embedding=_vec_at(0.99)),
    ]

    assert len(_dedupe(events)) == 1


# ---------------------------------------------------------------------------
# Missing embeddings
# ---------------------------------------------------------------------------


def test_event_without_summary_embedding_never_merged():
    events = [
        _event("a", summary_embedding=_reference()),
        _event("b", summary_embedding=None),
    ]

    assert len(_dedupe(events)) == 2


def test_two_events_without_embeddings_not_merged():
    """Absent vectors are unknown, not equal."""
    events = [_event("a", summary_embedding=None), _event("b", summary_embedding=None)]

    assert len(_dedupe(events)) == 2


# ---------------------------------------------------------------------------
# Merge quality
# ---------------------------------------------------------------------------


def test_merge_unions_source_attribution():
    events = [
        _event("a", candidates=["cand-1"], summary_embedding=_reference()),
        _event("b", candidates=["cand-2"], summary_embedding=_vec_at(0.98)),
    ]

    result = _dedupe(events)

    assert sorted(result[0].source_event_candidates) == ["cand-1", "cand-2"]


def test_merge_keeps_the_more_complete_record():
    sparse = _event("a", summary_embedding=_reference(), url=None)
    complete = _event(
        "b", summary_embedding=_vec_at(0.98), url="https://example.com",
        description="Full description", location="Salem, MA",
    )

    result = _dedupe([sparse, complete])

    assert result[0].url == "https://example.com"
    assert result[0].description == "Full description"


def test_merge_fills_gaps_from_the_secondary_record():
    a = _event("a", summary_embedding=_reference(), url="https://a.example")
    b = _event("b", summary_embedding=_vec_at(0.98), description="From B")

    result = _dedupe([a, b])

    assert result[0].url == "https://a.example"
    assert result[0].description == "From B"


def test_pass_one_merged_event_survives_intact():
    """A cluster already merged in Pass 1 arrives as one event; do not re-split it."""
    merged = _event("a", candidates=["c1", "c2", "c3"], summary_embedding=_reference())

    result = _dedupe([merged, _event("b", summary_embedding=_vec_at(0.10))])

    survivor = next(e for e in result if "c1" in e.source_event_candidates)
    assert sorted(survivor.source_event_candidates) == ["c1", "c2", "c3"]


def test_tags_and_embeddings_preserved_through_merge():
    a = _event("a", summary_embedding=_reference(), tags=[Tag(text="jazz")])
    a.tag_embeddings = [encode_vector([1.0, 2.0])]
    b = _event("b", summary_embedding=_vec_at(0.98))

    result = _dedupe([a, b])

    assert result[0].tags == [Tag(text="jazz")]
    assert result[0].tag_embeddings == [encode_vector([1.0, 2.0])]


# ---------------------------------------------------------------------------
# Degenerate input
# ---------------------------------------------------------------------------


def test_empty_list_returns_empty():
    assert _dedupe([]) == []


def test_single_event_returned_unchanged():
    event = _event("a", summary_embedding=_reference())

    assert _dedupe([event]) == [event]
