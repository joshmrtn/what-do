"""Contract every CandidateRepository implementation must satisfy.

Run against both the SQLite repository and the in-memory one, for the reasons in
`test_event_repository.py`.

Every field is asserted at a **non-default** value on purpose. A round-trip test
that leaves a field at its default passes against a column that does not exist,
which is exactly how `EventCandidate.timing` was silently dropped for weeks.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from src.models.event_candidate import EventCandidate
from src.models.tag import Tag
from src.storage.sqlite.connection import init_db
from src.storage.memory.candidates import InMemoryCandidateRepository
from src.storage.sqlite.candidates import SqliteCandidateRepository

_NOW = datetime(2026, 8, 11, 12, 0, tzinfo=timezone.utc)
_LOOKBACK = _NOW - timedelta(days=30)


@pytest.fixture(params=["sqlite", "memory"])
def repo(request, tmp_path):
    """One repository per implementation, so every test below runs twice."""
    if request.param == "sqlite":
        path = tmp_path / "candidates.db"
        init_db(path)
        return SqliteCandidateRepository(path)
    return InMemoryCandidateRepository()


def _candidate(candidate_id: str = "c1", **kwargs) -> EventCandidate:
    defaults = dict(
        id=candidate_id,
        source="northshorenightout_listing",
        source_type="northshorenightout",
        discovered_at=_NOW,
        title="Trivia",
        start_time=_NOW + timedelta(hours=7),
    )
    defaults.update(kwargs)
    return EventCandidate(**defaults)


def _in_window(repo) -> list[EventCandidate]:
    return repo.for_window(discovered_since=_LOOKBACK, starting_after=_NOW)


class TestRoundTrip:
    def test_every_field_survives_at_a_non_default_value(self, repo):
        repo.save(
            [
                _candidate(
                    url="https://example.test/e/1",
                    image_url="https://example.test/i/1.jpg",
                    raw_published_at=_NOW - timedelta(days=2),
                    description="Doors at seven.",
                    venue="The James",
                    location="Essex",
                    end_time=_NOW + timedelta(hours=10),
                    timing="all_day",
                    summary="Trivia at The James in Essex",
                    tags=[Tag(text="trivia", weight=1.0)],
                    metadata={"authored_tags": True},
                )
            ]
        )

        stored = _in_window(repo)[0]

        assert stored.url == "https://example.test/e/1"
        assert stored.image_url == "https://example.test/i/1.jpg"
        assert stored.raw_published_at == _NOW - timedelta(days=2)
        assert stored.description == "Doors at seven."
        assert stored.venue == "The James"
        assert stored.location == "Essex"
        assert stored.end_time == _NOW + timedelta(hours=10)
        assert stored.timing == "all_day"
        assert stored.summary == "Trivia at The James in Essex"
        assert [(t.text, t.weight) for t in stored.tags] == [("trivia", 1.0)]
        assert stored.metadata == {"authored_tags": True}

    def test_absent_optional_fields_read_back_as_defaults(self, repo):
        repo.save([_candidate()])

        stored = _in_window(repo)[0]

        assert (stored.summary, stored.tags, stored.metadata) == (None, [], {})
        assert stored.timing == "exact"

    def test_saving_nothing_does_not_clear_the_store(self, repo):
        repo.save([_candidate()])
        repo.save([])

        assert len(_in_window(repo)) == 1

    def test_an_empty_store_reads_back_empty(self, repo):
        assert _in_window(repo) == []


class TestReplacement:
    def test_refetching_the_same_id_replaces_it(self, repo):
        repo.save([_candidate(title="Trivia")])
        repo.save([_candidate(title="Trivia w/ Lee Wolf")])

        stored = _in_window(repo)
        assert len(stored) == 1
        assert stored[0].title == "Trivia w/ Lee Wolf"

    def test_other_candidates_are_untouched(self, repo):
        repo.save([_candidate("keep", title="Karaoke"), _candidate("change")])
        repo.save([_candidate("change", title="changed")])

        by_id = {c.id: c.title for c in _in_window(repo)}
        assert by_id == {"keep": "Karaoke", "change": "changed"}


class TestWindow:
    """The window is a union, because either filter alone starves a source."""

    def test_a_recently_discovered_candidate_with_no_start_is_kept(self, repo):
        """Social candidates carry no start_time, so a forward-only filter drops them."""
        repo.save([_candidate(start_time=None)])

        assert len(_in_window(repo)) == 1

    def test_an_upcoming_candidate_discovered_long_ago_is_kept(self, repo):
        """A discovery-only filter eventually drops events that are still upcoming."""
        repo.save(
            [_candidate(discovered_at=_NOW - timedelta(days=90),
                        start_time=_NOW + timedelta(days=5))]
        )

        assert len(_in_window(repo)) == 1

    def test_an_old_candidate_that_has_already_happened_is_dropped(self, repo):
        repo.save(
            [_candidate(discovered_at=_NOW - timedelta(days=90),
                        start_time=_NOW - timedelta(days=60))]
        )

        assert _in_window(repo) == []

    def test_results_are_ordered_by_discovery_then_id(self, repo):
        """Dedup picks a merge base partly on the order it sees, so it is fixed."""
        repo.save(
            [
                _candidate("b", discovered_at=_NOW),
                _candidate("a", discovered_at=_NOW),
                _candidate("early", discovered_at=_NOW - timedelta(days=1)),
            ]
        )

        assert [c.id for c in _in_window(repo)] == ["early", "a", "b"]
