"""Contract every EventRepository implementation must satisfy.

Run against both the SQLite repository and the in-memory one. The in-memory
implementation exists to be the single official fake: hand-written fakes drift
from the real contract silently, and a suite full of them stays green while
production breaks. Anything asserted here holds for both, so substituting the
fake in a stage's tests cannot quietly change the rules.

Behaviour that only a database can have — foreign keys, cascades, not rewriting
untouched rows — is *not* here. It lives in `test_sqlite_event_repository.py`,
because the alternative is a fake reimplementing referential integrity, and a
permissive fake is the drift problem wearing a different hat.
"""

from __future__ import annotations

import zoneinfo
from datetime import datetime, timezone

import pytest

from src.models.event import Event
from src.models.tag import Tag
from src.storage.sqlite.connection import init_db
from src.storage.memory.events import InMemoryEventRepository
from src.storage.sqlite.events import SqliteEventRepository
from src.utils.vectors import encode_vector

_TZ = zoneinfo.ZoneInfo("America/New_York")
_NOW = datetime(2026, 6, 15, 12, 0, tzinfo=timezone.utc)


@pytest.fixture(params=["sqlite", "memory"])
def repo(request, tmp_path):
    """One repository per implementation, so every test below runs twice."""
    if request.param == "sqlite":
        path = tmp_path / "events.db"
        init_db(path)
        return SqliteEventRepository(path)
    return InMemoryEventRepository()


def _event(event_id="e1", **kwargs) -> Event:
    defaults = dict(
        source_event_candidates=["c1", "c2"],
        source_type="apify",
        created_at=_NOW,
        updated_at=_NOW,
    )
    defaults.update(kwargs)
    return Event(event_id=event_id, **defaults)


def _full_event(event_id="e1") -> Event:
    event = _event(
        event_id,
        url="https://example.com/e",
        title="Karaoke Night",
        venue="Koto",
        description="A full description.",
        location="Salem, MA",
        start_time=datetime(2026, 8, 26, 20, 30, tzinfo=_TZ),
        end_time=datetime(2026, 8, 26, 23, 30, tzinfo=_TZ),
        tags=[Tag("karaoke", 1.0), Tag("bar", 0.2)],
        summary="A karaoke night at Koto.",
    )
    event.tag_embeddings = [encode_vector([1.0, 2.0, 3.0]), encode_vector([4.0, 5.0, 6.0])]
    event.summary_embedding = encode_vector([7.0, 8.0, 9.0])
    return event


def _only(repo) -> Event:
    events = repo.load_all()
    assert len(events) == 1
    return events[0]


# ---------------------------------------------------------------------------
# Round trip — the fields where a silent bug corrupts rather than raises
# ---------------------------------------------------------------------------


def test_round_trip_preserves_scalar_fields(repo):
    repo.save([_full_event()])

    loaded = _only(repo)

    assert loaded.event_id == "e1"
    assert loaded.title == "Karaoke Night"
    assert loaded.venue == "Koto"
    assert loaded.url == "https://example.com/e"
    assert loaded.description == "A full description."
    assert loaded.location == "Salem, MA"
    assert loaded.summary == "A karaoke night at Koto."
    assert loaded.source_type == "apify"


def test_round_trip_preserves_tag_order_and_weights(repo):
    """Position is what keeps a tag aligned with its weight.

    Reordering changes nothing visible and silently rescores the event, so the
    order is asserted as a sequence rather than as a set.
    """
    repo.save([_full_event()])

    assert [(t.text, t.weight) for t in _only(repo).tags] == [("karaoke", 1.0), ("bar", 0.2)]


def test_round_trip_preserves_vector_bytes_exactly(repo):
    """float32 is the invariant. Compared as bytes, because a float compare
    would pass on a vector quietly widened to float64."""
    event = _full_event()
    repo.save([event])

    loaded = _only(repo)

    assert loaded.tag_embeddings == event.tag_embeddings
    assert loaded.summary_embedding == event.summary_embedding


def test_round_trip_preserves_timezone_aware_times(repo):
    """A naive datetime reaching production has already broken this system once."""
    repo.save([_full_event()])

    loaded = _only(repo)

    assert loaded.start_time == datetime(2026, 8, 26, 20, 30, tzinfo=_TZ)
    assert loaded.end_time == datetime(2026, 8, 26, 23, 30, tzinfo=_TZ)
    assert loaded.start_time.utcoffset() is not None
    assert loaded.created_at == _NOW


def test_round_trip_preserves_the_callers_timestamps(repo):
    """The repository stores time, it does not decide it.

    If it ever stamps its own, it needs an injected clock — pinning this stops
    that arriving unnoticed.
    """
    created = datetime(2020, 1, 2, 3, 4, tzinfo=timezone.utc)
    updated = datetime(2021, 5, 6, 7, 8, tzinfo=timezone.utc)
    repo.save([_event(created_at=created, updated_at=updated)])

    loaded = _only(repo)

    assert loaded.created_at == created
    assert loaded.updated_at == updated


def test_round_trip_preserves_absent_optionals_as_none(repo):
    """None must not come back as an empty string — they mean different things."""
    repo.save([_event()])

    loaded = _only(repo)

    assert loaded.title is None
    assert loaded.venue is None
    assert loaded.url is None
    assert loaded.end_time is None
    assert loaded.summary is None
    assert loaded.summary_embedding is None


def test_round_trip_preserves_unicode(repo):
    """Source text carries accents and emoji, and text corruption is silent."""
    repo.save([_event(title="Café Sessions 🎷", venue="Grüner Löwe", tags=[Tag("jazz café", 1.0)])])

    loaded = _only(repo)

    assert loaded.title == "Café Sessions 🎷"
    assert loaded.venue == "Grüner Löwe"
    assert [t.text for t in loaded.tags] == ["jazz café"]


def test_event_with_no_tags_round_trips_as_an_empty_list(repo):
    repo.save([_event()])

    assert _only(repo).tags == []


def test_round_trip_preserves_source_candidates(repo):
    """Provenance: which raw candidates were merged into this event."""
    repo.save([_event(source_event_candidates=["c1", "c2"])])

    assert sorted(_only(repo).source_event_candidates) == ["c1", "c2"]


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------


def test_save_empty_list_does_not_clear_the_store(repo):
    """"Nothing to save" is never "delete everything"."""
    repo.save([_full_event()])

    repo.save([])

    assert len(repo.load_all()) == 1


def test_delete_empty_list_does_not_clear_the_store(repo):
    repo.save([_full_event()])

    repo.delete([])

    assert len(repo.load_all()) == 1


def test_save_one_is_equivalent_to_saving_a_list_of_one(repo, tmp_path):
    """Two write paths that disagree is a bug that only shows up in production."""
    event = _full_event()
    repo.save_one(event)

    loaded = _only(repo)

    assert loaded == event


def test_save_one_leaves_other_events_intact(repo):
    repo.save([_full_event("keep")])

    repo.save_one(_full_event("added"))

    assert {e.event_id for e in repo.load_all()} == {"keep", "added"}


def test_mutating_an_event_after_saving_does_not_reach_the_store(repo):
    """A store holds a copy, not a reference to the caller's object.

    SQLite gets this free by serialising. Without it the in-memory
    implementation would hand back the very object under test, and every
    round-trip assertion above would be comparing an object with itself.
    """
    event = _full_event()
    repo.save([event])

    event.title = "Changed after saving"
    event.tags.append(Tag("late-addition", 0.5))

    loaded = _only(repo)
    assert loaded.title == "Karaoke Night"
    assert [t.text for t in loaded.tags] == ["karaoke", "bar"]


def test_mutating_a_loaded_event_does_not_reach_the_store(repo):
    repo.save([_full_event()])

    loaded = _only(repo)
    loaded.title = "Changed after loading"

    assert _only(repo).title == "Karaoke Night"


def test_saving_the_same_event_twice_keeps_one_copy(repo):
    repo.save([_full_event()])
    repo.save([_full_event()])

    assert len(repo.load_all()) == 1


def test_resaving_an_event_replaces_its_tags(repo):
    """Tags are rows, not a column: re-saving must clear the old ones.

    Without that, an event that loses a tag keeps it forever and scores on a
    tag the extractor no longer believes in.
    """
    event = _full_event()
    repo.save([event])

    event.tags = [Tag("karaoke", 1.0)]
    event.tag_embeddings = [encode_vector([1.0, 2.0, 3.0])]
    repo.save([event])

    assert [t.text for t in _only(repo).tags] == ["karaoke"]


def test_saving_an_event_with_tags_but_no_vectors_is_allowed(repo):
    """The normal state between extraction and embedding.

    Extraction writes tags and the embedding stage fills vectors afterwards, so
    "no vectors at all" means unembedded, not corrupt.
    """
    event = _full_event()
    event.tag_embeddings = []

    repo.save([event])

    loaded = _only(repo)
    assert [t.text for t in loaded.tags] == ["karaoke", "bar"]
    assert loaded.tag_embeddings == []


def test_saving_a_partial_tag_vector_count_raises(repo):
    """Silently zipping unequal lists stores a subset and reports success.

    Distinct from the case above: *some* vectors but not one per tag cannot be
    explained by ordering, so the event is wrong upstream and storage swallowing
    it is how that stays invisible.
    """
    event = _full_event()
    event.tag_embeddings = [encode_vector([1.0, 2.0, 3.0])]

    with pytest.raises(ValueError, match="tag"):
        repo.save([event])


# ---------------------------------------------------------------------------
# Deleting
# ---------------------------------------------------------------------------


def test_delete_removes_only_the_named_events(repo):
    repo.save([_full_event("gone"), _full_event("kept")])

    repo.delete(["gone"])

    assert [e.event_id for e in repo.load_all()] == ["kept"]


def test_delete_accepts_more_ids_than_sqlite_can_bind(repo):
    """The corpus is already past 1,000 events.

    A query building one placeholder per id hits SQLITE_MAX_VARIABLE_NUMBER and
    fails on a night when enough events happen to be superseded at once.
    """
    events = [_event(f"e{n}") for n in range(1200)]
    repo.save(events)

    repo.delete([f"e{n}" for n in range(1200)])

    assert repo.load_all() == []


# ---------------------------------------------------------------------------
# Tag vectors
# ---------------------------------------------------------------------------


def test_tag_embeddings_returns_every_stored_vector_by_tag(repo):
    """Seeds the embedding stage's memo, so a tag embedded on any previous night
    is never embedded again."""
    repo.save([_full_event()])

    assert repo.tag_embeddings() == {
        "karaoke": encode_vector([1.0, 2.0, 3.0]),
        "bar": encode_vector([4.0, 5.0, 6.0]),
    }


def test_tag_embeddings_is_empty_before_anything_is_saved(repo):
    assert repo.tag_embeddings() == {}
