"""Unit tests for deduplication engine (Pass 1 — fuzzy, pre-embedding)."""

from __future__ import annotations

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import uuid

import pytest

from src.config import DeduplicationConfig
from src.models.event import Event
from src.normalization.deduplicator import DeduplicationEngine


_TZ = ZoneInfo("America/New_York")
_BASE_TIME = datetime(2025, 6, 15, 20, 0, 0, tzinfo=_TZ)
_DEFAULT_CFG = DeduplicationConfig(
    fuzzy_title_threshold=0.85,
    time_window_hours=2.0,
)


def _event(**kwargs) -> Event:
    now = datetime(2025, 6, 15, 12, 0, 0, tzinfo=_TZ)
    defaults = dict(
        event_id=str(uuid.uuid4()),
        source_event_candidates=[str(uuid.uuid4())],
        source_type="apify",
        title="Jazz Night at The Vault",
        venue="The Vault",
        start_time=_BASE_TIME,
        created_at=now,
        updated_at=now,
    )
    defaults.update(kwargs)
    return Event(**defaults)


def _dedup(events: list[Event], cfg: DeduplicationConfig = _DEFAULT_CFG) -> list[Event]:
    return DeduplicationEngine().deduplicate(events, cfg).events


# --- identical pair merges ---

def test_identical_title_venue_time_merges_to_one():
    a = _event(title="Jazz Night", venue="The Vault")
    b = _event(title="Jazz Night", venue="The Vault")
    result = _dedup([a, b])
    assert len(result) == 1


def test_merged_event_contains_both_source_candidate_ids():
    a = _event(title="Jazz Night", venue="The Vault",
               source_event_candidates=["cand-a"])
    b = _event(title="Jazz Night", venue="The Vault",
               source_event_candidates=["cand-b"])
    result = _dedup([a, b])
    assert set(result[0].source_event_candidates) == {"cand-a", "cand-b"}


# --- fuzzy title match ---

def test_similar_title_same_venue_within_time_window_merges():
    a = _event(title="Jazz Night at The Vault")
    b = _event(title="Jazz Night @ The Vault")  # minor variation
    result = _dedup([a, b])
    assert len(result) == 1


def test_very_different_titles_same_venue_not_merged():
    a = _event(title="Jazz Night", venue="The Vault")
    b = _event(title="Trivia Tuesday", venue="The Vault")
    result = _dedup([a, b])
    assert len(result) == 2


# --- venue criterion ---

def test_same_title_different_venue_not_merged():
    a = _event(title="Jazz Night", venue="The Vault")
    b = _event(title="Jazz Night", venue="The Anchor")
    result = _dedup([a, b])
    assert len(result) == 2


def test_same_title_both_venue_none_merges():
    a = _event(title="Jazz Night", venue=None)
    b = _event(title="Jazz Night", venue=None)
    result = _dedup([a, b])
    assert len(result) == 1


def test_same_title_one_venue_none_not_merged():
    a = _event(title="Jazz Night", venue="The Vault")
    b = _event(title="Jazz Night", venue=None)
    result = _dedup([a, b])
    assert len(result) == 2


# --- time window criterion ---

def test_same_title_venue_within_window_merges():
    a = _event(start_time=_BASE_TIME)
    b = _event(start_time=_BASE_TIME + timedelta(hours=1))
    result = _dedup([a, b])
    assert len(result) == 1


def test_same_title_venue_outside_window_not_merged():
    a = _event(start_time=_BASE_TIME)
    b = _event(start_time=_BASE_TIME + timedelta(hours=3))
    result = _dedup([a, b])
    assert len(result) == 2


def test_same_title_venue_exactly_at_window_boundary_merges():
    a = _event(start_time=_BASE_TIME)
    b = _event(start_time=_BASE_TIME + timedelta(hours=2))
    result = _dedup([a, b])
    assert len(result) == 1


def test_both_start_times_none_time_criterion_passes():
    a = _event(title="Jazz Night", venue="The Vault", start_time=None)
    b = _event(title="Jazz Night", venue="The Vault", start_time=None)
    result = _dedup([a, b])
    assert len(result) == 1


def test_one_start_time_none_not_merged():
    a = _event(title="Jazz Night", venue="The Vault", start_time=_BASE_TIME)
    b = _event(title="Jazz Night", venue="The Vault", start_time=None)
    result = _dedup([a, b])
    assert len(result) == 2


# --- both titles None ---

def test_both_titles_none_same_venue_time_merges():
    a = _event(title=None, venue="The Vault")
    b = _event(title=None, venue="The Vault")
    result = _dedup([a, b])
    assert len(result) == 1


def test_one_title_none_other_present_not_merged():
    a = _event(title="Jazz Night", venue="The Vault")
    b = _event(title=None, venue="The Vault")
    result = _dedup([a, b])
    assert len(result) == 2


# --- most-complete-wins merge ---

def test_most_complete_record_wins():
    """Event with venue populated wins over one without."""
    complete = _event(
        title="Jazz Night",
        venue="The Vault",
        description="Great show",
        source_event_candidates=["cand-complete"],
    )
    sparse = _event(
        title="Jazz Night",
        venue="The Vault",
        description=None,
        source_event_candidates=["cand-sparse"],
    )
    result = _dedup([sparse, complete])
    assert result[0].description == "Great show"


def test_null_fields_from_loser_merged_into_winner():
    """Fields missing in winner but present in secondary are adopted."""
    winner = _event(
        title="Jazz Night",
        venue="The Vault",
        description="Great show",
        location=None,
        source_event_candidates=["cand-winner"],
    )
    secondary = _event(
        title="Jazz Night",
        venue="The Vault",
        description=None,
        location="Salem, MA",
        source_event_candidates=["cand-secondary"],
    )
    result = _dedup([winner, secondary])
    assert result[0].location == "Salem, MA"
    assert result[0].description == "Great show"


def test_tiebreak_earlier_discovered_at_wins():
    """When both candidates have equal completeness, earlier created_at is base."""
    tz = _TZ
    earlier_time = datetime(2025, 6, 14, 10, 0, 0, tzinfo=tz)
    later_time = datetime(2025, 6, 15, 10, 0, 0, tzinfo=tz)
    early = _event(
        title="Jazz Night",
        description="from early",
        created_at=earlier_time,
        updated_at=earlier_time,
        source_event_candidates=["cand-early"],
    )
    late = _event(
        title="Jazz Night",
        description="from late",
        created_at=later_time,
        updated_at=later_time,
        source_event_candidates=["cand-late"],
    )
    result = _dedup([late, early])
    assert result[0].description == "from early"


# --- threshold reads from config ---

def test_dedup_threshold_respected():
    """With a very high threshold, near-identical titles should not merge."""
    strict_cfg = DeduplicationConfig(fuzzy_title_threshold=0.99, time_window_hours=2.0)
    a = _event(title="Jazz Night at The Vault")
    b = _event(title="Jazz Night @ The Vault")
    result = _dedup([a, b], cfg=strict_cfg)
    assert len(result) == 2


# --- three-way dedup ---

def test_three_duplicates_merge_to_one():
    a = _event(title="Jazz Night", venue="The Vault", source_event_candidates=["c1"])
    b = _event(title="Jazz Night", venue="The Vault", source_event_candidates=["c2"])
    c = _event(title="Jazz Night", venue="The Vault", source_event_candidates=["c3"])
    result = _dedup([a, b, c])
    assert len(result) == 1
    assert set(result[0].source_event_candidates) == {"c1", "c2", "c3"}


def test_mixed_batch_two_unique_one_duplicate_pair():
    dup_a = _event(title="Jazz Night", venue="The Vault", source_event_candidates=["c1"])
    dup_b = _event(title="Jazz Night", venue="The Vault", source_event_candidates=["c2"])
    unique = _event(title="Trivia Night", venue="The Anchor", source_event_candidates=["c3"])
    result = _dedup([dup_a, dup_b, unique])
    assert len(result) == 2


# --- single event passthrough ---

def test_single_event_returned_unchanged():
    event = _event()
    result = _dedup([event])
    assert len(result) == 1
    assert result[0].event_id == event.event_id


def test_empty_list_returns_empty():
    assert _dedup([]) == []


class TestDecisionsAreReported:
    """A merge that records nothing cannot be explained, and a comparison that
    records nothing cannot be learned from.

    The scores both passes compute were discarded inside a `bool`-returning
    predicate. They are the labelled data a future dedup model trains on, and
    the *rejections* are most of it: measured on the live corpus, the whole
    database holds two merges and 1,629 considered-and-rejected pairs.
    """

    @staticmethod
    def _decisions(events, cfg=_DEFAULT_CFG):
        return DeduplicationEngine().deduplicate(events, cfg).decisions

    def test_a_merged_pair_is_reported_with_its_score(self):
        a = _event(title="Jazz Night", source_event_candidates=["cand-a"])
        b = _event(title="Jazz Night", source_event_candidates=["cand-b"])

        decisions = self._decisions([a, b])

        assert len(decisions) == 1
        assert decisions[0].verdict == "merged"
        assert decisions[0].score == pytest.approx(1.0)

    def test_a_rejected_pair_is_reported_with_its_score(self):
        """The blind spot a row-shaped design cannot express: two events were
        compared and judged *different*. Most of the training signal is here."""
        a = _event(title="Jazz Night", source_event_candidates=["cand-a"])
        b = _event(title="Poetry Slam", source_event_candidates=["cand-b"])

        decisions = self._decisions([a, b])

        assert len(decisions) == 1
        assert decisions[0].verdict == "distinct"
        assert 0.0 < decisions[0].score < _DEFAULT_CFG.fuzzy_title_threshold

    def test_a_pair_failing_the_structural_guards_is_not_reported(self):
        """Never compared, so there is nothing to record. The guards discard
        99.87% of all pairs — recording them would be recording nothing."""
        a = _event(venue="The Vault", source_event_candidates=["cand-a"])
        b = _event(venue="Somewhere Else", source_event_candidates=["cand-b"])

        assert self._decisions([a, b]) == []

    def test_a_transitive_cluster_reports_every_pair_it_compared(self):
        """A~B and B~C collapse all three, and A~C is compared on the way.

        One `merge_similarity` on the surviving row cannot express three
        comparisons with three different scores — which is why the decision,
        not the row, is the unit of record.
        """
        a = _event(title="Jazz Night", source_event_candidates=["cand-a"])
        b = _event(title="Jazz Nite", source_event_candidates=["cand-b"])
        c = _event(title="Jazz Night!", source_event_candidates=["cand-c"])

        decisions = self._decisions([a, b, c])

        assert len(decisions) == 3
        assert len({d.score for d in decisions}) > 1, "each pair scored on its own"

    def test_the_pair_is_identified_the_same_whichever_order_it_arrives(self):
        """One row per pair, so `(a, b)` and `(b, a)` must be the same pair."""
        a = _event(title="Jazz Night", source_event_candidates=["cand-a"])
        b = _event(title="Jazz Night", source_event_candidates=["cand-b"])

        forward = self._decisions([a, b])[0]
        backward = self._decisions([b, a])[0]

        assert (forward.record_a, forward.record_b) == (backward.record_a, backward.record_b)
        assert forward.record_a < forward.record_b

    def test_pass_one_names_itself_and_keys_on_candidates(self):
        """Pass 1's events are minted per run, so keying on their ids would
        make every pair new every night and accumulate nothing. At this point
        each event carries exactly one candidate, and candidate ids are stable."""
        a = _event(title="Jazz Night", source_event_candidates=["cand-a"])
        b = _event(title="Jazz Night", source_event_candidates=["cand-b"])

        decision = self._decisions([a, b])[0]

        assert decision.pass_name == "fuzzy"
        assert decision.record_kind == "candidate"
        assert {decision.record_a, decision.record_b} == {"cand-a", "cand-b"}

    def test_a_record_with_no_stable_id_is_merged_but_not_recorded(self):
        """A decision keyed on an id that will not exist tomorrow is worse than
        no decision — it would never match again and would accumulate forever."""
        a = _event(title="Jazz Night", source_event_candidates=[])
        b = _event(title="Jazz Night", source_event_candidates=["cand-b"])

        result = DeduplicationEngine().deduplicate([a, b], _DEFAULT_CFG)

        assert len(result.events) == 1, "the merge still happens"
        assert result.decisions == []

    def test_the_merged_events_are_unchanged_by_reporting(self):
        """This adds an output. It must not alter one."""
        a = _event(title="Jazz Night", source_event_candidates=["cand-a"])
        b = _event(title="Jazz Night", source_event_candidates=["cand-b"])

        result = DeduplicationEngine().deduplicate([a, b], _DEFAULT_CFG)

        assert len(result.events) == 1
        assert set(result.events[0].source_event_candidates) == {"cand-a", "cand-b"}


class TestTheComparedTextIsFingerprinted:
    """`event_candidates` is written with INSERT OR REPLACE, so a re-fetched
    listing overwrites the stored row in place (#27).

    A person cleaning this data months from now would then be reading text that
    is not what dedup compared, with nothing saying so. The hash does not
    recover the original — it makes the substitution *detectable*.
    """

    @staticmethod
    def _decisions(events, cfg=_DEFAULT_CFG):
        return DeduplicationEngine().deduplicate(events, cfg).decisions

    def test_each_side_records_a_hash_of_what_was_compared(self):
        a = _event(title="Jazz Night", source_event_candidates=["cand-a"])
        b = _event(title="Jazz Night", source_event_candidates=["cand-b"])

        decision = self._decisions([a, b])[0]

        assert decision.content_hash_a
        assert decision.content_hash_a == decision.content_hash_b, "same text, same hash"

    def test_different_text_fingerprints_differently(self):
        a = _event(title="Jazz Night", source_event_candidates=["cand-a"])
        b = _event(title="Jazz Nite", source_event_candidates=["cand-b"])

        decision = self._decisions([a, b])[0]

        assert decision.content_hash_a != decision.content_hash_b

    def test_an_edited_listing_changes_the_fingerprint(self):
        """The whole point: re-running after a source edits its listing must
        produce a hash that no longer matches the stored decision."""
        before = self._decisions([
            _event(title="Jazz Night", description="Trio.", source_event_candidates=["cand-a"]),
            _event(title="Jazz Night", source_event_candidates=["cand-b"]),
        ])[0]
        after = self._decisions([
            _event(title="Jazz Night", description="Quartet.", source_event_candidates=["cand-a"]),
            _event(title="Jazz Night", source_event_candidates=["cand-b"]),
        ])[0]

        assert before.content_hash_a != after.content_hash_a
