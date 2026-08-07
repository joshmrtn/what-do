"""Unit tests for reconciling freshly normalized events against stored ones."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from src.models.event import Event
from src.models.tag import Tag
from src.normalization.reconcile import reconcile

NOW = datetime(2025, 6, 15, 12, 0, 0, tzinfo=timezone.utc)
EARLIER = NOW - timedelta(days=3)


def _event(event_id: str, candidates: list[str], **overrides) -> Event:
    fields = {
        "event_id": event_id,
        "source_event_candidates": candidates,
        "source_type": "apify",
        "created_at": NOW,
        "updated_at": NOW,
    }
    fields.update(overrides)
    return Event(**fields)


def _enriched(event_id: str, candidates: list[str], **overrides) -> Event:
    """A stored event carrying everything the expensive stages produce."""
    fields = {
        "created_at": EARLIER,
        "updated_at": EARLIER,
        "tags": [Tag(text="karaoke", weight=1.0)],
        "summary": "Karaoke night at The Vault",
        "setting": "indoor",
        "tag_embeddings": [b"\x00\x01"],
        "summary_embedding": b"\x02\x03",
        "astronomical_data": {"sunset": "2025-06-16T20:15:00"},
        "weather": {"sampled_hour": 20},
    }
    fields.update(overrides)
    return _event(event_id, candidates, **fields)


def test_no_stored_events_returns_fresh_untouched():
    fresh = [_event("new-1", ["c1"])]

    result = reconcile(fresh, [])

    assert result.events == fresh
    assert result.stale_event_ids == []


def test_unmatched_fresh_event_keeps_its_own_id():
    result = reconcile([_event("new-1", ["c1"])], [_enriched("old-1", ["c9"])])

    assert result.events[0].event_id == "new-1"
    assert result.events[0].tags == []


def test_single_match_adopts_the_stored_id():
    result = reconcile([_event("new-1", ["c1"])], [_enriched("old-1", ["c1"])])

    assert result.events[0].event_id == "old-1"


def test_single_match_carries_forward_extraction_output():
    """The whole point: tags survive, so ExtractionStage skips and re-runs nothing."""
    result = reconcile([_event("new-1", ["c1"])], [_enriched("old-1", ["c1"])])

    carried = result.events[0]
    assert [t.text for t in carried.tags] == ["karaoke"]
    assert carried.summary == "Karaoke night at The Vault"
    assert carried.setting == "indoor"


def test_single_match_carries_forward_embeddings():
    result = reconcile([_event("new-1", ["c1"])], [_enriched("old-1", ["c1"])])

    assert result.events[0].tag_embeddings == [b"\x00\x01"]
    assert result.events[0].summary_embedding == b"\x02\x03"


def test_single_match_carries_forward_astronomical_data():
    """Deterministic from date and location, so it can never go stale."""
    result = reconcile([_event("new-1", ["c1"])], [_enriched("old-1", ["c1"])])

    assert result.events[0].astronomical_data == {"sunset": "2025-06-16T20:15:00"}


def test_single_match_never_carries_forward_weather():
    """A carried forecast would score an event forever on the day it was found."""
    result = reconcile([_event("new-1", ["c1"])], [_enriched("old-1", ["c1"])])

    assert result.events[0].weather is None


def test_single_match_keeps_the_stored_creation_time():
    """created_at records when the event was first seen, not when it was last read."""
    result = reconcile([_event("new-1", ["c1"])], [_enriched("old-1", ["c1"])])

    assert result.events[0].created_at == EARLIER
    assert result.events[0].updated_at == NOW


def test_single_match_keeps_freshly_scraped_content():
    """The source is authoritative for content; storage only supplies identity."""
    fresh = _event("new-1", ["c1"], title="Karaoke Night (moved to 9pm)", venue="The Vault")
    result = reconcile([fresh], [_enriched("old-1", ["c1"], title="Karaoke Night")])

    assert result.events[0].title == "Karaoke Night (moved to 9pm)"


def test_match_is_found_on_any_shared_candidate():
    result = reconcile([_event("new-1", ["c2"])], [_enriched("old-1", ["c1", "c2"])])

    assert result.events[0].event_id == "old-1"


def test_grown_cluster_adopts_the_richest_match():
    """Two stored events merging into one is live today, not hypothetical."""
    rich = _enriched("old-rich", ["c1"])
    thin = _event("old-thin", ["c2"], created_at=EARLIER, updated_at=EARLIER)

    result = reconcile([_event("new-1", ["c1", "c2"])], [rich, thin])

    assert result.events[0].event_id == "old-rich"
    assert [t.text for t in result.events[0].tags] == ["karaoke"]


def test_grown_cluster_reports_the_loser_as_stale():
    """Without the delete, the loser lingers forever as a duplicate in the output."""
    rich = _enriched("old-rich", ["c1"])
    thin = _event("old-thin", ["c2"], created_at=EARLIER, updated_at=EARLIER)

    result = reconcile([_event("new-1", ["c1", "c2"])], [rich, thin])

    assert result.stale_event_ids == ["old-thin"]


def test_equally_rich_matches_break_the_tie_on_creation_time():
    older = _enriched("old-a", ["c1"], created_at=EARLIER - timedelta(days=5))
    newer = _enriched("old-b", ["c2"], created_at=EARLIER)

    result = reconcile([_event("new-1", ["c1", "c2"])], [older, newer])

    assert result.events[0].event_id == "old-a"
    assert result.stale_event_ids == ["old-b"]


def test_identically_aged_matches_break_the_tie_on_id():
    """Ties must resolve the same way on every run, never on dict ordering."""
    first = _enriched("old-a", ["c1"])
    second = _enriched("old-b", ["c2"])

    forward = reconcile([_event("new-1", ["c1", "c2"])], [first, second])
    reversed_ = reconcile([_event("new-1", ["c1", "c2"])], [second, first])

    assert forward.events[0].event_id == reversed_.events[0].event_id == "old-a"


def test_stored_event_with_no_fresh_counterpart_is_not_stale():
    """Its candidates fell out of the window; the event itself must survive."""
    result = reconcile([_event("new-1", ["c1"])], [_enriched("old-1", ["c9"])])

    assert result.stale_event_ids == []


def test_a_split_cluster_is_merged_back_onto_its_stored_event():
    """The stored event is the record of a semantic verdict already paid for.

    Dedup pass 1 is fuzzy and splits what pass 2 merged on a previous run. If
    the second fresh event is emitted as new instead, it carries no tags,
    re-extracts at roughly three minutes, and is saved as another row — every
    night, without bound.
    """
    stored = _enriched("old-1", ["c1", "c2"])

    result = reconcile([_event("new-a", ["c1"]), _event("new-b", ["c2"])], [stored])

    assert [e.event_id for e in result.events] == ["old-1"]
    assert sorted(result.events[0].source_event_candidates) == ["c1", "c2"]


def test_a_merged_split_keeps_the_stored_enrichment():
    """The whole point: nothing re-extracts because the tags came along."""
    stored = _enriched("old-1", ["c1", "c2"])

    result = reconcile([_event("new-a", ["c1"]), _event("new-b", ["c2"])], [stored])

    assert result.events[0].tags != []
    assert result.events[0].summary_embedding == b"\x02\x03"


def test_a_merged_split_takes_content_from_both_sides():
    """Which fresh event is 'first' is arbitrary, so neither may lose its content."""
    stored = _enriched("old-1", ["c1", "c2"])
    fresh = [
        _event("new-a", ["c1"], title="Eras Tour Tribute"),
        _event("new-b", ["c2"], venue="Kowloon"),
    ]

    result = reconcile(fresh, [stored])

    assert result.events[0].title == "Eras Tour Tribute"
    assert result.events[0].venue == "Kowloon"


def test_two_stored_events_sharing_a_candidate_are_both_seen():
    """Indexing candidate -> one event hides every owner but the last.

    A hidden owner is never reported stale, so it lingers as a duplicate for
    the life of the database.
    """
    rich = _enriched("old-rich", ["c1"])
    thin = _event("old-thin", ["c1"])

    result = reconcile([_event("new-a", ["c1"])], [rich, thin])

    assert result.events[0].event_id == "old-rich"
    assert result.stale_event_ids == ["old-thin"]


def test_fresh_order_is_preserved():
    fresh = [_event("new-1", ["c1"]), _event("new-2", ["c2"]), _event("new-3", ["c3"])]

    result = reconcile(fresh, [_enriched("old-2", ["c2"])])

    assert [e.event_id for e in result.events] == ["new-1", "old-2", "new-3"]


def test_reconcile_does_not_mutate_its_inputs():
    fresh = _event("new-1", ["c1"])
    stored = _enriched("old-1", ["c1"])

    reconcile([fresh], [stored])

    assert fresh.event_id == "new-1"
    assert fresh.tags == []
    assert stored.weather == {"sampled_hour": 20}
