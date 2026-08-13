"""Reloading an event skips the work already paid for.

The round-trip and delete cases that used to live here are in
`test_event_repository.py`, where they run against both implementations —
a contract belongs to the protocol, not to one implementation of it.

These two are not contract tests. They are the claim that persisting is
worth anything at all: a reloaded event carries the hashes that make
extraction and embedding skip it, so a nightly batch pays for each event
once rather than every night. That spans storage and two stages, which is
why it cannot be asserted from inside either.
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock
import zoneinfo

import pytest

from src.models.event import Event
from src.models.tag import Tag
from src.processing.extraction_stage import ExtractionStage, extraction_input_hash
from src.scoring.embedding_stage import EmbeddingStage, embedding_input_hash
from src.storage.sqlite.connection import init_db
from src.storage.events import load_events, save_events
from src.utils.vectors import encode_vector

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
        metadata={"listing_category": "Music"},
    )
    event.tag_embeddings = [encode_vector([1.0, 2.0, 3.0]), encode_vector([4.0, 5.0, 6.0])]
    event.summary_embedding = encode_vector([7.0, 8.0, 9.0])
    event.extraction_input_hash = extraction_input_hash(event)
    event.embedding_input_hash = embedding_input_hash(event)
    return event


# ---------------------------------------------------------------------------
# Round trip
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Sparse events
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Save semantics
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Reload skips completed work — the point of persisting at all
# ---------------------------------------------------------------------------


def test_reloaded_event_bypasses_extraction(db):

    save_events([_full_event()], db)
    provider = MagicMock()

    ExtractionStage(provider, None, MagicMock()).process(load_events(db))

    provider.extract.assert_not_called()


def test_reloaded_event_bypasses_embedding(db):


    save_events([_full_event()], db)
    provider = MagicMock()

    EmbeddingStage(provider, MagicMock()).process(load_events(db))

    provider.embed.assert_not_called()


