"""Contract every EntityRepository implementation must satisfy.

Run against both the SQLite repository and the in-memory one.

The interesting behaviour here is **upsert with accumulation**. Every other
repository writes a run's output wholesale; this one increments a counter and
appends to a list, and a fake that does not accumulate identically is not a fake
worth substituting.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from src.models.candidate_entity import ACTIVE, DISCARDED, PROBATIONARY
from src.storage.sqlite.connection import init_db
from src.storage.memory.entities import InMemoryEntityRepository
from src.storage.sqlite.entities import SqliteEntityRepository

_NOW = datetime(2026, 8, 11, 12, 0, tzinfo=timezone.utc)
_LATER = datetime(2026, 8, 12, 12, 0, tzinfo=timezone.utc)


@pytest.fixture(params=["sqlite", "memory"])
def repo(request, tmp_path):
    """One repository per implementation, so every test below runs twice."""
    if request.param == "sqlite":
        path = tmp_path / "entities.db"
        init_db(path)
        return SqliteEntityRepository(path)
    return InMemoryEntityRepository()


def _mention(repo, handle="@jazzclub", source="@seed", depth=1, context="great show", now=_NOW):
    repo.record_mention(
        handle=handle, source_handle=source, depth=depth, context=context, now=now
    )


class TestSeedHandles:
    def test_a_seed_is_stored_active_at_depth_zero(self, repo):
        repo.mark_seeds_active(["@seed"], now=_NOW)

        assert repo.active_handles() == ["@seed"]

    def test_seeding_twice_does_not_duplicate(self, repo):
        repo.mark_seeds_active(["@seed"], now=_NOW)
        repo.mark_seeds_active(["@seed"], now=_LATER)

        assert repo.active_handles() == ["@seed"]

    def test_a_discovered_handle_named_as_a_seed_is_promoted_in_place(self, repo):
        """Adding a handle to seeds.yaml activates the row already there.

        In place matters: the mention evidence is why the handle was worth
        keeping, and recreating the row would silently discard it. Asserting
        only that the handle went active passes either way, which is what a
        mutation of the memory implementation showed.
        """
        _mention(repo, handle="@jazzclub", source="@a")
        _mention(repo, handle="@jazzclub", source="@b")
        discovered = repo.unclassified()[0]

        repo.mark_seeds_active(["@jazzclub"], now=_LATER)

        assert repo.active_handles() == ["@jazzclub"]
        promoted = repo.by_handle("@jazzclub")
        assert promoted.entity_id == discovered.entity_id, "the row was recreated"
        assert promoted.mention_count == 2
        assert promoted.mention_sources == ["@a", "@b"]

    def test_active_handles_are_alphabetical(self, repo):
        repo.mark_seeds_active(["@zed", "@alpha", "@mid"], now=_NOW)

        assert repo.active_handles() == ["@alpha", "@mid", "@zed"]

    def test_nothing_active_reads_back_empty(self, repo):
        assert repo.active_handles() == []


class TestRecordMention:
    """Upsert with accumulation — the part a naive fake gets wrong."""

    def test_a_first_sighting_starts_probationary_with_one_mention(self, repo):
        _mention(repo)

        entity = repo.unclassified()[0]
        assert entity.handle == "@jazzclub"
        assert entity.state == PROBATIONARY
        assert entity.mention_count == 1
        assert entity.mention_sources == ["@seed"]
        assert entity.depth == 1

    def test_a_new_source_increments_the_count(self, repo):
        _mention(repo, source="@one")
        _mention(repo, source="@two")

        entity = repo.unclassified()[0]
        assert entity.mention_count == 2
        assert entity.mention_sources == ["@one", "@two"]

    def test_the_same_source_twice_counts_once(self, repo):
        """Otherwise one chatty account promotes its own friends."""
        _mention(repo, source="@one")
        _mention(repo, source="@one")

        assert repo.unclassified()[0].mention_count == 1

    def test_the_first_context_wins(self, repo):
        _mention(repo, source="@one", context="the original sighting")
        _mention(repo, source="@two", context="a later one")

        assert repo.unclassified()[0].discovery_context == "the original sighting"

    def test_a_first_context_is_filled_when_there_was_none(self, repo):
        _mention(repo, source="@one", context=None)
        _mention(repo, source="@two", context="now we have one")

        assert repo.unclassified()[0].discovery_context == "now we have one"

    def test_mentioning_an_active_handle_does_not_demote_it(self, repo):
        repo.mark_seeds_active(["@seed"], now=_NOW)

        _mention(repo, handle="@seed", source="@other")

        assert repo.active_handles() == ["@seed"]


class TestClassification:
    def test_only_unclassified_probationary_handles_are_offered(self, repo):
        _mention(repo, handle="@one", source="@a")
        _mention(repo, handle="@two", source="@a")
        repo.classify(
            repo.unclassified()[0].entity_id,
            classification="venue",
            state=PROBATIONARY,
            now=_NOW,
        )

        assert [e.handle for e in repo.unclassified()] == ["@two"]

    def test_a_person_is_discarded(self, repo):
        _mention(repo)
        entity_id = repo.unclassified()[0].entity_id

        repo.classify(entity_id, classification="person", state=DISCARDED, now=_NOW)

        assert repo.unclassified() == []
        assert repo.awaiting_promotion() == []

    def test_an_active_handle_is_never_offered_for_classification(self, repo):
        repo.mark_seeds_active(["@seed"], now=_NOW)

        assert repo.unclassified() == []


class TestPromotion:
    def _venue(self, repo, handle="@jazzclub", sources=("@a",)):
        for source in sources:
            _mention(repo, handle=handle, source=source)
        entity = next(e for e in repo.unclassified() if e.handle == handle)
        repo.classify(
            entity.entity_id, classification="venue", state=PROBATIONARY, now=_NOW
        )
        return entity.entity_id

    def test_a_classified_venue_awaits_promotion(self, repo):
        self._venue(repo)

        assert [e.handle for e in repo.awaiting_promotion()] == ["@jazzclub"]

    def test_an_unclassified_handle_does_not(self, repo):
        _mention(repo)

        assert repo.awaiting_promotion() == []

    def test_the_evidence_travels_with_it(self, repo):
        """Promotion needs the count and the sources to decide."""
        self._venue(repo, sources=("@a", "@b"))

        entity = repo.awaiting_promotion()[0]
        assert entity.mention_count == 2
        assert entity.mention_sources == ["@a", "@b"]

    def test_activating_it_makes_it_a_source(self, repo):
        entity_id = self._venue(repo)

        repo.activate(entity_id, now=_LATER)

        assert repo.active_handles() == ["@jazzclub"]
        assert repo.awaiting_promotion() == []
