"""Unit tests for the Event model."""

from datetime import datetime, timezone

import pytest

from src.models.event import Event


def _now() -> datetime:
    return datetime(2025, 6, 15, 12, 0, 0, tzinfo=timezone.utc)


def test_event_construction_minimal():
    """Event can be constructed with only required fields."""
    event = Event(
        event_id="abc-123",
        source_event_candidates=["cand-1"],
        source_type="apify",
        created_at=_now(),
        updated_at=_now(),
    )
    assert event.event_id == "abc-123"
    assert event.source_type == "apify"


def test_event_source_event_candidates_is_list_of_strings():
    event = Event(
        event_id="abc-123",
        source_event_candidates=["cand-1", "cand-2"],
        source_type="apify",
        created_at=_now(),
        updated_at=_now(),
    )
    assert event.source_event_candidates == ["cand-1", "cand-2"]


def test_event_optional_fields_default_none():
    event = Event(
        event_id="abc-123",
        source_event_candidates=["cand-1"],
        source_type="apify",
        created_at=_now(),
        updated_at=_now(),
    )
    assert event.url is None
    assert event.image_url is None
    assert event.title is None
    assert event.venue is None
    assert event.description is None
    assert event.location is None
    assert event.start_time is None
    assert event.end_time is None
    assert event.summary is None
    assert event.summary_embedding is None
    assert event.weather is None
    assert event.astronomical_data is None


def test_event_tags_default_empty_list():
    event = Event(
        event_id="abc-123",
        source_event_candidates=["cand-1"],
        source_type="apify",
        created_at=_now(),
        updated_at=_now(),
    )
    assert event.tags == []


def test_event_tag_embeddings_default_empty_list():
    event = Event(
        event_id="abc-123",
        source_event_candidates=["cand-1"],
        source_type="apify",
        created_at=_now(),
        updated_at=_now(),
    )
    assert event.tag_embeddings == []


def test_event_metadata_defaults_empty_dict():
    event = Event(
        event_id="abc-123",
        source_event_candidates=["cand-1"],
        source_type="apify",
        created_at=_now(),
        updated_at=_now(),
    )
    assert event.metadata == {}


def test_event_metadata_not_shared_across_instances():
    """Default metadata dicts must be independent per instance."""
    a = Event(
        event_id="a",
        source_event_candidates=[],
        source_type="apify",
        created_at=_now(),
        updated_at=_now(),
    )
    b = Event(
        event_id="b",
        source_event_candidates=[],
        source_type="apify",
        created_at=_now(),
        updated_at=_now(),
    )
    a.metadata["x"] = 1
    assert "x" not in b.metadata


def test_event_tags_not_shared_across_instances():
    """Default tags lists must be independent per instance."""
    a = Event(
        event_id="a",
        source_event_candidates=[],
        source_type="apify",
        created_at=_now(),
        updated_at=_now(),
    )
    b = Event(
        event_id="b",
        source_event_candidates=[],
        source_type="apify",
        created_at=_now(),
        updated_at=_now(),
    )
    a.tags.append("music")
    assert b.tags == []


def test_event_full_construction():
    """Event with all fields set round-trips correctly."""
    start = datetime(2025, 6, 15, 20, 0, 0, tzinfo=timezone.utc)
    end = datetime(2025, 6, 15, 23, 0, 0, tzinfo=timezone.utc)
    event = Event(
        event_id="full-1",
        source_event_candidates=["c1", "c2"],
        source_type="cinema_veezi",
        url="https://example.com/event",
        image_url="https://example.com/img.jpg",
        title="Jazz Night",
        venue="The Vault",
        description="A great jazz show.",
        location="Salem, MA",
        start_time=start,
        end_time=end,
        tags=["jazz", "live music"],
        summary="An evening of jazz at The Vault.",
        tag_embeddings=[b"fake-bytes"],
        summary_embedding=b"more-fake-bytes",
        weather={"condition": "clear"},
        astronomical_data={"sunset": "20:30"},
        metadata={"source": "test"},
        created_at=_now(),
        updated_at=_now(),
    )
    assert event.title == "Jazz Night"
    assert event.venue == "The Vault"
    assert event.tags == ["jazz", "live music"]
    assert event.start_time == start
    assert len(event.source_event_candidates) == 2
