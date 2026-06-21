"""Unit tests for the EventCandidate data model."""

from __future__ import annotations

from datetime import datetime, timezone


def test_event_candidate_instantiates_with_required_fields():
    from src.models.event_candidate import EventCandidate

    now = datetime.now(timezone.utc)
    ec = EventCandidate(
        id="abc123",
        source="@testvenue",
        source_type="apify",
        discovered_at=now,
    )
    assert ec.id == "abc123"
    assert ec.source == "@testvenue"
    assert ec.source_type == "apify"
    assert ec.discovered_at == now


def test_event_candidate_optional_fields_default_none():
    from src.models.event_candidate import EventCandidate

    ec = EventCandidate(
        id="x",
        source="s",
        source_type="apify",
        discovered_at=datetime.now(timezone.utc),
    )
    assert ec.url is None
    assert ec.image_url is None
    assert ec.raw_published_at is None
    assert ec.title is None
    assert ec.description is None
    assert ec.venue is None
    assert ec.location is None
    assert ec.start_time is None
    assert ec.end_time is None


def test_event_candidate_accepts_all_fields():
    from src.models.event_candidate import EventCandidate

    now = datetime.now(timezone.utc)
    ec = EventCandidate(
        id="full",
        source="@venue",
        source_type="apify",
        url="https://example.com/post/1",
        image_url="https://example.com/img.jpg",
        raw_published_at=now,
        title="Jazz Night",
        description="Live jazz every Friday",
        venue="The Vault",
        location="Salem, MA",
        start_time=now,
        end_time=now,
        discovered_at=now,
    )
    assert ec.title == "Jazz Night"
    assert ec.raw_published_at == now
