"""Storage for dedup decisions — the training corpus, and its integrity.

This table is read by a person months from now deciding whether a merge was
right, and later by whatever learns from their answer. So the round trip is
asserted at non-default values throughout: a column that exists but is never
written reads back as its default, and every test passes.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from src.normalization.decision_sampling import (
    STRATUM_MERGED,
    STRATUM_NEAR_MISS,
    STRATUM_SAMPLED,
    SampledDecision,
)
from src.normalization.deduplicator import MergeDecision
from src.storage.sqlite.connection import connect, init_db
from src.storage.sqlite.dedup_decisions import SqliteDedupDecisionRepository

_NOW = datetime(2026, 8, 14, 6, 0, tzinfo=timezone.utc)


@pytest.fixture
def db(tmp_path):
    path = tmp_path / "decisions.db"
    init_db(path)
    conn = connect(path)
    conn.execute(
        "INSERT INTO run_history (id, started_at) VALUES (?, ?)",
        ("run-1", _NOW.isoformat()),
    )
    conn.commit()
    conn.close()
    return path


def _sampled(
    *,
    a="evt-a",
    b="evt-b",
    score=0.93,
    verdict="merged",
    stratum=STRATUM_MERGED,
    denominator=1,
    pass_name="semantic",
    record_kind="event",
) -> SampledDecision:
    return SampledDecision(
        decision=MergeDecision(
            pass_name=pass_name,
            record_kind=record_kind,
            record_a=a,
            record_b=b,
            score=score,
            verdict=verdict,
            # Derived from the id so that a pair reordered on write is caught
            # carrying the wrong side's fingerprint.
            content_hash_a=f"hash-of-{a}",
            content_hash_b=f"hash-of-{b}",
        ),
        stratum=stratum,
        sample_denominator=denominator,
    )


class TestTheRoundTrip:
    def test_every_column_survives_at_a_non_default_value(self, db):
        """The guard the four-part change demands. A field whose column was
        never added reads back as its default and nothing throws."""
        repo = SqliteDedupDecisionRepository(db)
        repo.save(
            [
                _sampled(
                    a="evt-b",  # deliberately unsorted on the way in
                    b="evt-a",
                    score=0.8125,
                    verdict="distinct",
                    stratum=STRATUM_SAMPLED,
                    denominator=10,
                    pass_name="fuzzy",
                    record_kind="candidate",
                )
            ],
            run_id="run-1",
            now=_NOW,
        )

        stored = repo.load_all()

        assert len(stored) == 1
        row = stored[0]
        assert row.pass_name == "fuzzy"
        assert row.record_kind == "candidate"
        assert row.score == pytest.approx(0.8125)
        assert row.verdict == "distinct"
        assert row.stratum == STRATUM_SAMPLED
        assert row.sample_denominator == 10
        assert row.content_hash_a == "hash-of-evt-a"
        assert row.content_hash_b == "hash-of-evt-b"
        assert row.run_id == "run-1"

    def test_a_pair_is_one_row_however_often_it_is_seen(self, db):
        """Volume grows with distinct comparisons, not with nights. The same
        pair is compared again every run it survives."""
        repo = SqliteDedupDecisionRepository(db)
        repo.save([_sampled(score=0.93)], run_id="run-1", now=_NOW)
        repo.save([_sampled(score=0.97)], run_id="run-1", now=_NOW)

        stored = repo.load_all()

        assert len(stored) == 1
        assert stored[0].score == pytest.approx(0.97), "the latest verdict wins"

    def test_the_same_pair_under_two_passes_is_two_rows(self, db):
        """Pass 1 and Pass 2 judge on different evidence and may disagree —
        which is itself worth keeping."""
        repo = SqliteDedupDecisionRepository(db)
        repo.save(
            [
                _sampled(pass_name="fuzzy", record_kind="candidate"),
                _sampled(pass_name="semantic", record_kind="event"),
            ],
            run_id="run-1",
            now=_NOW,
        )

        assert len(repo.load_all()) == 2

    def test_saving_nothing_is_not_an_error(self, db):
        SqliteDedupDecisionRepository(db).save([], run_id="run-1", now=_NOW)

        assert SqliteDedupDecisionRepository(db).load_all() == []


class TestTheDecisionTiesBackToItsRecords:
    def test_the_pair_is_stored_in_sorted_order(self, db):
        """One identity per pair, whichever order it was compared in."""
        repo = SqliteDedupDecisionRepository(db)
        repo.save([_sampled(a="evt-z", b="evt-a")], run_id="run-1", now=_NOW)

        row = repo.load_all()[0]

        assert (row.record_a, row.record_b) == ("evt-a", "evt-z")
        assert (row.content_hash_a, row.content_hash_b) == (
            "hash-of-evt-a",
            "hash-of-evt-z",
        ), "the fingerprints must travel with their own side"

    def test_a_pair_written_both_ways_round_is_still_one_row(self, db):
        """The primary key rests on pair order. A reversed write that skipped
        normalisation would quietly become a second row for one pair, and the
        table's whole premise is one row per pair."""
        repo = SqliteDedupDecisionRepository(db)
        repo.save([_sampled(a="evt-a", b="evt-z")], run_id="run-1", now=_NOW)
        repo.save([_sampled(a="evt-z", b="evt-a")], run_id="run-1", now=_NOW)

        assert len(repo.load_all()) == 1

    def test_the_run_is_recoverable_so_the_thresholds_are_too(self, db):
        """A verdict is a function of thresholds that will be tuned. Without
        the run, a retuned threshold silently reinterprets every old label."""
        repo = SqliteDedupDecisionRepository(db)
        repo.save([_sampled()], run_id="run-1", now=_NOW)

        conn = connect(db)
        try:
            joined = conn.execute(
                "SELECT r.started_at FROM dedup_decisions d "
                "JOIN run_history r ON r.id = d.run_id"
            ).fetchone()
        finally:
            conn.close()

        assert joined is not None
