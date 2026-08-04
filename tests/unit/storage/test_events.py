"""Unit tests for event persistence and reload."""

from __future__ import annotations

import zoneinfo
from datetime import datetime, timezone

import pytest

from src.models.event import Event
from src.models.tag import Tag
from src.storage.db import init_db
from src.storage.events import load_events, save_events
from src.utils.vectors import decode_vector, encode_vector

_TZ = zoneinfo.ZoneInfo("America/New_York")
_NOW = datetime(2026, 6, 15, 12, 0, tzinfo=timezone.utc)


@pytest.fixture
def db(tmp_path):
    path = tmp_path / "events.db"
    init_db(path)
    return path


def _event(event_id="e1", **kwargs) -> Event:
    defaults = dict(
        source_event_candidates=["c1", "c2"],
        source_type="apify",
        created_at=_NOW,
        updated_at=_NOW,
    )
    defaults.update(kwargs)
    return Event(event_id=event_id, **defaults)


def _full_event() -> Event:
    event = _event(
        url="https://example.com/e",
        image_url="https://example.com/i.jpg",
        title="Karaoke Night",
        venue="Koto",
        description="A full description.",
        location="Salem, MA",
        start_time=datetime(2026, 8, 26, 20, 30, tzinfo=_TZ),
        end_time=datetime(2026, 8, 26, 23, 30, tzinfo=_TZ),
        tags=[Tag("karaoke", 1.0), Tag("bar", 0.2)],
        summary="A karaoke night at Koto.",
        weather={"temperature_f": 70.0, "condition": "clear"},
        astronomical_data={"sunset": "20:15"},
        metadata={"llm_extraction_failed": False},
    )
    event.tag_embeddings = [encode_vector([1.0, 2.0, 3.0]), encode_vector([4.0, 5.0, 6.0])]
    event.summary_embedding = encode_vector([7.0, 8.0, 9.0])
    return event


# ---------------------------------------------------------------------------
# Round trip
# ---------------------------------------------------------------------------


def test_round_trip_preserves_scalar_fields(db):
    save_events([_full_event()], db)

    loaded = load_events(db)[0]

    assert loaded.event_id == "e1"
    assert loaded.title == "Karaoke Night"
    assert loaded.venue == "Koto"
    assert loaded.description == "A full description."
    assert loaded.location == "Salem, MA"
    assert loaded.url == "https://example.com/e"
    assert loaded.image_url == "https://example.com/i.jpg"
    assert loaded.summary == "A karaoke night at Koto."
    assert loaded.source_type == "apify"


def test_round_trip_preserves_weighted_tags(db):
    save_events([_full_event()], db)

    assert load_events(db)[0].tags == [Tag("karaoke", 1.0), Tag("bar", 0.2)]


def test_round_trip_preserves_tag_embeddings(db):
    save_events([_full_event()], db)

    loaded = load_events(db)[0]

    assert len(loaded.tag_embeddings) == 2
    assert decode_vector(loaded.tag_embeddings[0]) == pytest.approx([1.0, 2.0, 3.0])
    assert decode_vector(loaded.tag_embeddings[1]) == pytest.approx([4.0, 5.0, 6.0])


def test_round_trip_preserves_summary_embedding(db):
    save_events([_full_event()], db)

    loaded = load_events(db)[0]

    assert decode_vector(loaded.summary_embedding) == pytest.approx([7.0, 8.0, 9.0])


def test_round_trip_preserves_timezone_aware_times(db):
    save_events([_full_event()], db)

    loaded = load_events(db)[0]

    assert loaded.start_time.utcoffset() is not None
    assert loaded.start_time == datetime(2026, 8, 26, 20, 30, tzinfo=_TZ)
    assert loaded.end_time == datetime(2026, 8, 26, 23, 30, tzinfo=_TZ)


def test_round_trip_preserves_json_fields(db):
    save_events([_full_event()], db)

    loaded = load_events(db)[0]

    assert loaded.source_event_candidates == ["c1", "c2"]
    assert loaded.weather == {"temperature_f": 70.0, "condition": "clear"}
    assert loaded.astronomical_data == {"sunset": "20:15"}
    assert loaded.metadata == {"llm_extraction_failed": False}


# ---------------------------------------------------------------------------
# Sparse events
# ---------------------------------------------------------------------------


def test_minimal_event_round_trips(db):
    save_events([_event()], db)

    loaded = load_events(db)[0]

    assert loaded.title is None
    assert loaded.tags == []
    assert loaded.tag_embeddings == []
    assert loaded.summary_embedding is None
    assert loaded.start_time is None


def test_tags_without_embeddings_round_trip(db):
    """Extraction may have run while embedding has not."""
    save_events([_event(tags=[Tag("karaoke"), Tag("bar", 0.2)])], db)

    loaded = load_events(db)[0]

    assert len(loaded.tags) == 2
    assert loaded.tag_embeddings == []


# ---------------------------------------------------------------------------
# Save semantics
# ---------------------------------------------------------------------------


def test_saving_twice_updates_rather_than_duplicates(db):
    event = _full_event()
    save_events([event], db)
    event.title = "Updated Title"
    save_events([event], db)

    loaded = load_events(db)

    assert len(loaded) == 1
    assert loaded[0].title == "Updated Title"


def test_save_persists_every_event(db):
    save_events([_event("a"), _event("b"), _event("c")], db)

    assert sorted(e.event_id for e in load_events(db)) == ["a", "b", "c"]


def test_saving_empty_list_is_a_no_op(db):
    save_events([], db)

    assert load_events(db) == []


def test_similarity_is_not_persisted(db):
    """Scores are derived and cheap to recompute; recommendations own them."""
    from src.scoring.similarity import SimilarityResult

    event = _full_event()
    event.similarity = SimilarityResult(base_score=0.8, match="yes")
    save_events([event], db)

    assert load_events(db)[0].similarity is None


# ---------------------------------------------------------------------------
# Reload skips completed work — the point of persisting at all
# ---------------------------------------------------------------------------


def test_reloaded_event_bypasses_extraction(db):
    from src.processing.extraction_stage import ExtractionStage
    from unittest.mock import MagicMock

    save_events([_full_event()], db)
    provider = MagicMock()

    ExtractionStage(provider, None, MagicMock()).process(load_events(db))

    provider.extract.assert_not_called()


def test_reloaded_event_bypasses_embedding(db):
    from unittest.mock import MagicMock

    from src.scoring.embedding_stage import EmbeddingStage

    save_events([_full_event()], db)
    provider = MagicMock()

    EmbeddingStage(provider, MagicMock()).process(load_events(db))

    provider.embed.assert_not_called()


def test_round_trip_preserves_setting(db):
    save_events([_event(setting="outdoor")], db)
    assert load_events(db)[0].setting == "outdoor"


def test_setting_defaults_to_unknown(db):
    save_events([_event()], db)
    assert load_events(db)[0].setting == "unknown"
