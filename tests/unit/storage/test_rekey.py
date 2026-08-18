"""Unit tests for re-keying a source onto content-derived candidate ids.

This operation is run twice over the same code: once by hand against the live
database, and thereafter by the churn latch, unattended, at 02:00. Everything it
touches is irreversible, so the assertions here are deliberately heavier than
the module's size suggests.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from src.ingestion.candidate_id import derive_content_id
from src.models.event_candidate import EventCandidate
from src.storage.rekey import RekeyFailed, rekey_to_content_ids
from src.storage.sqlite.candidates import SqliteCandidateRepository
from src.storage.sqlite.connection import connect, init_db

NOW = datetime(2026, 8, 18, 6, 0, tzinfo=timezone.utc)
START = datetime(2026, 8, 22, 23, 0, tzinfo=timezone.utc)
SOURCE = "northshorenightout"


@pytest.fixture
def db(tmp_path):
    path = tmp_path / "event_hub.db"
    init_db(path)
    return path


def _candidate(cid: str, **overrides) -> EventCandidate:
    fields: dict = {
        "id": cid,
        "source": SOURCE,
        "source_type": "northshorenightout",
        "title": "Isabel Stover",
        "venue": "The Joy Nest",
        "start_time": START,
        "discovered_at": NOW,
        "last_seen_at": NOW,
    }
    fields.update(overrides)
    return EventCandidate(**fields)


def _store(db, *candidates: EventCandidate) -> None:
    SqliteCandidateRepository(db).save(list(candidates))


def _link(db, event_id: str, *candidate_ids: str) -> None:
    """Attach candidates to an event, creating the event row if needed."""
    with connect(db) as conn:
        conn.execute(
            "INSERT OR IGNORE INTO events (id, title, source_type, created_at, "
            "updated_at) VALUES (?, ?, ?, ?, ?)",
            (event_id, "Isabel Stover", "northshorenightout", NOW.isoformat(),
             NOW.isoformat()),
        )
        conn.executemany(
            "INSERT OR IGNORE INTO event_source_candidates (event_id, candidate_id) "
            "VALUES (?, ?)",
            [(event_id, cid) for cid in candidate_ids],
        )
        conn.commit()


def _ids(db, source: str = SOURCE) -> set[str]:
    with connect(db) as conn:
        return {
            row[0]
            for row in conn.execute(
                "SELECT id FROM event_candidates WHERE source = ?", (source,)
            )
        }


def _expected_id() -> str:
    return derive_content_id(
        source=SOURCE, title="Isabel Stover", venue="The Joy Nest", start=START
    )


class TestCollapsingTheDuplicates:
    def test_rows_for_one_listing_become_one_row(self, db):
        _store(db, _candidate("uid-1"), _candidate("uid-2"), _candidate("uid-3"))

        rekey_to_content_ids(db, source=SOURCE)

        assert _ids(db) == {_expected_id()}

    def test_the_surviving_row_carries_the_content_id(self, db):
        _store(db, _candidate("uid-1"))

        rekey_to_content_ids(db, source=SOURCE)

        assert _ids(db) == {_expected_id()}

    def test_distinct_listings_are_not_merged(self, db):
        _store(
            db,
            _candidate("uid-1"),
            _candidate("uid-2", title="Someone Else"),
            _candidate("uid-3", start_time=START + timedelta(days=1)),
        )

        rekey_to_content_ids(db, source=SOURCE)

        assert len(_ids(db)) == 3

    def test_another_source_is_untouched(self, db):
        _store(db, _candidate("uid-1"), _candidate("veezi:38750", source="cinemasalem"))

        rekey_to_content_ids(db, source=SOURCE)

        assert _ids(db, "cinemasalem") == {"veezi:38750"}

    def test_the_outcome_reports_what_it_did(self, db):
        _store(db, _candidate("uid-1"), _candidate("uid-2"), _candidate("uid-3"))

        outcome = rekey_to_content_ids(db, source=SOURCE)

        assert outcome.candidates_before == 3
        assert outcome.candidates_after == 1
        assert outcome.absorbed == 2


class TestWhatTheSurvivingRowKeeps:
    """A collapsed group is one listing seen many times. The survivor has to
    read as *first seen then, last seen now* — which for this feed has never
    been true of any row, because no id was ever seen twice."""

    def test_it_keeps_the_earliest_first_sighting(self, db):
        _store(
            db,
            _candidate("uid-late", discovered_at=NOW),
            _candidate("uid-early", discovered_at=NOW - timedelta(days=5)),
        )

        rekey_to_content_ids(db, source=SOURCE)

        with connect(db) as conn:
            discovered = conn.execute(
                "SELECT discovered_at FROM event_candidates"
            ).fetchone()[0]
        assert discovered == (NOW - timedelta(days=5)).isoformat()

    def test_it_keeps_the_latest_sighting(self, db):
        _store(
            db,
            _candidate("uid-a", discovered_at=NOW - timedelta(days=5),
                       last_seen_at=NOW - timedelta(days=5)),
            _candidate("uid-b", discovered_at=NOW, last_seen_at=NOW),
        )

        rekey_to_content_ids(db, source=SOURCE)

        with connect(db) as conn:
            last_seen = conn.execute(
                "SELECT last_seen_at FROM event_candidates"
            ).fetchone()[0]
        assert last_seen == NOW.isoformat()

    def test_first_and_last_sighting_now_differ(self, db):
        """The repair, stated as the property it restores. Every row in this
        feed has `last_seen_at == discovered_at` today, because a re-minted id
        is never recognised as a second sighting."""
        _store(
            db,
            _candidate("uid-a", discovered_at=NOW - timedelta(days=5),
                       last_seen_at=NOW - timedelta(days=5)),
            _candidate("uid-b", discovered_at=NOW, last_seen_at=NOW),
        )

        rekey_to_content_ids(db, source=SOURCE)

        with connect(db) as conn:
            discovered, last_seen = conn.execute(
                "SELECT discovered_at, last_seen_at FROM event_candidates"
            ).fetchone()
        assert discovered < last_seen


class TestTheEventsKeepTheirCandidates:
    def test_a_link_follows_its_candidate_to_the_new_id(self, db):
        _store(db, _candidate("uid-1"))
        _link(db, "event-1", "uid-1")

        rekey_to_content_ids(db, source=SOURCE)

        with connect(db) as conn:
            links = conn.execute(
                "SELECT candidate_id FROM event_source_candidates"
            ).fetchall()
        assert [row[0] for row in links] == [_expected_id()]

    def test_an_event_claiming_the_whole_group_ends_with_one_link(self, db):
        """Reconcile matches a stored event by shared candidate id, so an event
        that loses every link becomes unreachable and is minted afresh next run.
        Collapsing three links to one is right; collapsing them to none is the
        duplicate-event disaster."""
        _store(db, _candidate("uid-1"), _candidate("uid-2"), _candidate("uid-3"))
        _link(db, "event-1", "uid-1", "uid-2", "uid-3")

        rekey_to_content_ids(db, source=SOURCE)

        with connect(db) as conn:
            count = conn.execute(
                "SELECT COUNT(*) FROM event_source_candidates WHERE event_id = ?",
                ("event-1",),
            ).fetchone()[0]
        assert count == 1

    def test_no_event_is_left_with_nothing(self, db):
        _store(db, _candidate("uid-1"), _candidate("uid-2", title="Someone Else"))
        _link(db, "event-1", "uid-1")
        _link(db, "event-2", "uid-2")

        rekey_to_content_ids(db, source=SOURCE)

        with connect(db) as conn:
            orphans = conn.execute(
                "SELECT COUNT(*) FROM events e WHERE NOT EXISTS ("
                "  SELECT 1 FROM event_source_candidates l WHERE l.event_id = e.id)"
            ).fetchone()[0]
        assert orphans == 0

    def test_two_events_landing_on_one_candidate_are_reported(self, db):
        """Left for reconcile, which merges them — but never silently."""
        _store(db, _candidate("uid-1"), _candidate("uid-2"))
        _link(db, "event-1", "uid-1")
        _link(db, "event-2", "uid-2")

        outcome = rekey_to_content_ids(db, source=SOURCE)

        assert outcome.candidates_shared_by_several_events == 1


class TestTheVersionHistoryIsRecovered:
    """860 version rows across 156 listings hold 210 distinct contents. Those
    are real upstream edits this feed has never been able to record, because a
    new id every night means a new candidate rather than a new version."""

    def _versions(self, db) -> list[tuple[str, str]]:
        with connect(db) as conn:
            return conn.execute(
                "SELECT candidate_id, content_hash FROM candidate_versions "
                "ORDER BY content_hash"
            ).fetchall()

    def test_identical_content_collapses_to_one_version(self, db):
        _store(db, _candidate("uid-1"), _candidate("uid-2"))

        rekey_to_content_ids(db, source=SOURCE)

        assert len(self._versions(db)) == 1

    def test_an_edit_survives_as_a_second_version(self, db):
        _store(
            db,
            _candidate("uid-1", description="As published"),
            _candidate("uid-2", description="Corrected later"),
        )

        rekey_to_content_ids(db, source=SOURCE)

        versions = self._versions(db)
        assert len(versions) == 2
        assert {row[0] for row in versions} == {_expected_id()}

    def test_a_repeated_publication_keeps_when_it_first_appeared(self, db):
        """Rows sharing a content hash are one publication seen again, so the
        version's `observed_at` is its birth, not its latest sighting. Keeping
        the later one would date every edit to whenever we last looked."""
        _store(
            db,
            _candidate("uid-old", discovered_at=NOW - timedelta(days=5),
                       last_seen_at=NOW - timedelta(days=5)),
            _candidate("uid-new", discovered_at=NOW, last_seen_at=NOW),
        )

        rekey_to_content_ids(db, source=SOURCE)

        with connect(db) as conn:
            observed = conn.execute(
                "SELECT observed_at FROM candidate_versions"
            ).fetchone()[0]
        assert observed == (NOW - timedelta(days=5)).isoformat()

    def test_a_version_never_points_at_a_deleted_candidate(self, db):
        _store(db, _candidate("uid-1"), _candidate("uid-2"))

        rekey_to_content_ids(db, source=SOURCE)

        with connect(db) as conn:
            dangling = conn.execute(
                "SELECT COUNT(*) FROM candidate_versions v WHERE NOT EXISTS ("
                "  SELECT 1 FROM event_candidates c WHERE c.id = v.candidate_id)"
            ).fetchone()[0]
        assert dangling == 0


class TestTheDedupCorpus:
    """`dedup_decisions` is training data. Once the ids collapse, a pair whose
    sides were two rows of one listing becomes a comparison of a candidate
    against itself — never a judgement about anything, and 2192 of the live
    corpus's 2843 rows."""

    def _decision(self, db, a: str, b: str, verdict: str = "merged") -> None:
        with connect(db) as conn:
            conn.execute(
                "INSERT INTO run_history (id, started_at, outcome) VALUES "
                "(?, ?, ?) ON CONFLICT DO NOTHING",
                ("run-1", NOW.isoformat(), "success"),
            )
            conn.execute(
                "INSERT INTO dedup_decisions (pass_name, record_kind, record_a, "
                "record_b, score, verdict, stratum, sample_denominator, "
                "content_hash_a, content_hash_b, run_id, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                ("fuzzy", "candidate", a, b, 0.99, verdict, "merged", 1,
                 f"hash-{a}", f"hash-{b}", "run-1", NOW.isoformat()),
            )
            conn.commit()

    def _pairs(self, db) -> list[tuple[str, str]]:
        with connect(db) as conn:
            return conn.execute(
                "SELECT record_a, record_b FROM dedup_decisions"
            ).fetchall()

    def test_a_pair_that_became_a_self_comparison_is_removed(self, db):
        _store(db, _candidate("uid-1"), _candidate("uid-2"))
        self._decision(db, "uid-1", "uid-2")

        outcome = rekey_to_content_ids(db, source=SOURCE)

        assert self._pairs(db) == []
        assert outcome.self_pairs_removed == 1

    def test_a_pair_between_distinct_listings_survives_re_keyed(self, db):
        _store(db, _candidate("uid-1"), _candidate("uid-2", title="Someone Else"))
        self._decision(db, "uid-1", "uid-2", verdict="distinct")

        outcome = rekey_to_content_ids(db, source=SOURCE)

        assert len(self._pairs(db)) == 1
        assert "uid-1" not in self._pairs(db)[0]
        assert outcome.decisions_rekeyed == 1

    def test_a_pair_touching_another_source_keeps_that_side(self, db):
        _store(db, _candidate("uid-1"), _candidate("other:1", source="cinemasalem"))
        self._decision(db, "uid-1", "other:1", verdict="distinct")

        rekey_to_content_ids(db, source=SOURCE)

        assert "other:1" in self._pairs(db)[0]

    def test_a_pair_touching_neither_side_is_left_alone(self, db):
        _store(
            db,
            _candidate("a:1", source="cinemasalem"),
            _candidate("a:2", source="cinemasalem", title="Someone Else"),
        )
        self._decision(db, "a:1", "a:2", verdict="distinct")

        outcome = rekey_to_content_ids(db, source=SOURCE)

        assert self._pairs(db) == [("a:1", "a:2")]
        assert outcome.decisions_rekeyed == 0

    def test_two_pairs_collapsing_onto_one_identity_do_not_collide(self, db):
        """The primary key is `(pass, a, b)`, and re-keying can land two rows on
        one identity in either order. Normalising in the writer is what stops
        the second insert failing — or worse, both surviving as mirror images."""
        _store(
            db,
            _candidate("uid-1"),
            _candidate("uid-2"),
            _candidate("uid-3", title="Someone Else"),
        )
        self._decision(db, "uid-1", "uid-3", verdict="distinct")
        self._decision(db, "uid-3", "uid-2", verdict="distinct")

        rekey_to_content_ids(db, source=SOURCE)

        pairs = self._pairs(db)
        assert len(pairs) == 1
        assert pairs[0][0] < pairs[0][1]


class TestWhichPublicationSurvives:
    """The group holds one listing as it was published on several nights. The
    survivor must carry the *most recent* of them: the earlier ones are stale,
    and until the next fetch overwrites it the stored row is what the pipeline
    reads."""

    def test_the_latest_publication_is_the_one_kept(self, db):
        _store(
            db,
            _candidate("uid-old", description="As first published",
                       discovered_at=NOW - timedelta(days=5),
                       last_seen_at=NOW - timedelta(days=5)),
            _candidate("uid-new", description="Corrected later",
                       discovered_at=NOW, last_seen_at=NOW),
        )

        rekey_to_content_ids(db, source=SOURCE)

        with connect(db) as conn:
            description = conn.execute(
                "SELECT description FROM event_candidates"
            ).fetchone()[0]
        assert description == "Corrected later"

    def test_the_choice_does_not_depend_on_insertion_order(self, db):
        """Two rows seen on the same night still have to resolve the same way
        every run, or a re-key would flip the stored content at random."""
        _store(
            db,
            _candidate("uid-b", description="B"),
            _candidate("uid-a", description="A"),
        )

        rekey_to_content_ids(db, source=SOURCE)

        with connect(db) as conn:
            first = conn.execute("SELECT description FROM event_candidates").fetchone()[0]

        other = db.parent / "again.db"
        init_db(other)
        _store(other, _candidate("uid-a", description="A"),
               _candidate("uid-b", description="B"))
        rekey_to_content_ids(other, source=SOURCE)

        with connect(other) as conn:
            second = conn.execute("SELECT description FROM event_candidates").fetchone()[0]

        assert first == second


class TestTheVerificationRefusesToCommit:
    """Checked while a rollback is still possible. The `event_scores` rebuild
    shipped a broken database by verifying after the commit."""

    def test_an_event_left_with_no_candidate_aborts(self, db):
        _store(db, _candidate("uid-1"))
        _link(db, "event-1", "uid-1")
        # An event whose only candidate belongs to nothing this re-key touches,
        # and which therefore cannot survive it.
        _link(db, "event-2", "missing-candidate")
        with connect(db) as conn:
            conn.execute(
                "DELETE FROM event_source_candidates WHERE candidate_id = ?",
                ("missing-candidate",),
            )
            conn.commit()

        with pytest.raises(RekeyFailed, match="no candidate"):
            rekey_to_content_ids(db, source=SOURCE)

    def test_a_refused_re_key_writes_nothing(self, db):
        _store(db, _candidate("uid-1"))
        _link(db, "event-1", "uid-1")
        _link(db, "event-2", "missing-candidate")
        with connect(db) as conn:
            conn.execute(
                "DELETE FROM event_source_candidates WHERE candidate_id = ?",
                ("missing-candidate",),
            )
            conn.commit()

        with pytest.raises(RekeyFailed):
            rekey_to_content_ids(db, source=SOURCE)

        assert _ids(db) == {"uid-1"}
