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
    return repo.for_window(seen_since=_LOOKBACK, starting_after=_NOW)


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

    def test_datetimes_come_back_timezone_aware(self, repo):
        """SQLite has no datetime type, so awareness is the storage layer's job.

        A naive timestamp reaching the pipeline is what killed the first live
        fetch, and comparing one against an aware clock raises rather than
        quietly misfiling.
        """
        repo.save([_candidate(start_time=_NOW + timedelta(days=1))])

        loaded = repo.for_window(seen_since=_LOOKBACK, starting_after=_NOW)[0]
        assert loaded.discovered_at.tzinfo is not None
        assert loaded.start_time is not None and loaded.start_time.tzinfo is not None

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


class TestFirstAndLastSeen:
    """Two timestamps, because one field cannot answer two questions.

    `discovered_at` is provenance — when we first met this listing. `last_seen_at`
    is currency — whether a source is still publishing it. A single restamped
    field silently means the second while being read as the first, which is #27.
    """

    def test_a_first_sighting_records_the_same_instant_for_both(self, repo):
        """An adapter observing a listing has met it and seen it at once."""
        repo.save([_candidate(discovered_at=_NOW)])

        stored = _in_window(repo)[0]

        assert stored.discovered_at == _NOW
        assert stored.last_seen_at == _NOW

    def test_refetching_moves_last_seen_and_leaves_first_discovery_alone(self, repo):
        """The whole point: a republished listing is current, not newly found."""
        later = _NOW + timedelta(days=3)
        repo.save([_candidate(discovered_at=_NOW)])
        repo.save([_candidate(discovered_at=later)])

        stored = _in_window(repo)[0]

        assert stored.discovered_at == _NOW
        assert stored.last_seen_at == later

    def test_last_seen_survives_a_round_trip_at_a_non_default_value(self, repo):
        """Set apart from `discovered_at`, or the column need not exist to pass."""
        repo.save([_candidate(discovered_at=_NOW, last_seen_at=_NOW + timedelta(days=1))])

        stored = _in_window(repo)[0]

        assert stored.last_seen_at == _NOW + timedelta(days=1)

    def test_last_seen_comes_back_timezone_aware(self, repo):
        repo.save([_candidate()])

        assert _in_window(repo)[0].last_seen_at.tzinfo is not None


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
    """The window splits on what is *known* about a candidate.

    A dated candidate is in scope while its event is still to come, however long
    ago we found it. An undated one is in scope while we are still seeing it
    published, because that is the only evidence there is. A missing start is a
    gap in what we know, not evidence about when — the same reading
    `_scope_filter` gives an undated event.

    The arms were previously a union, which let recent *discovery* alone reload a
    candidate whose event finished a week ago: 649 of 2124 on 2026-08-14, each
    one costing normalization, dedup, enrichment and embedding every night (#26).
    """

    def test_an_upcoming_candidate_is_kept(self, repo):
        repo.save([_candidate(start_time=_NOW + timedelta(days=2))])

        assert len(_in_window(repo)) == 1

    def test_a_candidate_whose_event_is_over_is_dropped_however_recently_seen(self, repo):
        """The whole of #26. Recent sighting says the listing is current; it says
        nothing about whether the event has already happened."""
        repo.save(
            [_candidate(discovered_at=_NOW, last_seen_at=_NOW,
                        start_time=_NOW - timedelta(hours=6))]
        )

        assert _in_window(repo) == []

    def test_an_upcoming_candidate_first_seen_before_the_lookback_is_kept(self, repo):
        """Guards against making recent sighting a *requirement*, which is the
        obvious fix and the wrong one: it drops calendar events that are still
        to come simply because we found them a while back."""
        repo.save(
            [_candidate(discovered_at=_NOW - timedelta(days=90),
                        last_seen_at=_NOW - timedelta(days=90),
                        start_time=_NOW + timedelta(days=5))]
        )

        assert len(_in_window(repo)) == 1

    def test_a_candidate_starting_exactly_at_the_bound_is_kept(self, repo):
        """Inclusive: an event beginning this instant has not happened yet."""
        repo.save([_candidate(discovered_at=_NOW - timedelta(days=45), start_time=_NOW)])

        assert len(_in_window(repo)) == 1

    def test_an_undated_candidate_seen_recently_is_kept(self, repo):
        """Social candidates carry no start_time, so a forward-only filter would
        drop every one of them."""
        repo.save([_candidate(start_time=None)])

        assert len(_in_window(repo)) == 1

    def test_an_undated_candidate_not_seen_lately_is_dropped(self, repo):
        """The undated arm expires; otherwise every social post lives forever."""
        repo.save(
            [_candidate(discovered_at=_NOW - timedelta(days=45),
                        last_seen_at=_NOW - timedelta(days=45), start_time=None)]
        )

        assert _in_window(repo) == []

    def test_an_undated_candidate_still_being_published_is_kept(self, repo):
        """Reads `last_seen_at`, not `discovered_at` — a distinction that did not
        exist until the raw layer stopped restamping one field for both (#27).
        A listing a source is still publishing is current however long ago we
        first met it, and first-seen would have expired this one."""
        repo.save(
            [_candidate(discovered_at=_NOW - timedelta(days=90),
                        last_seen_at=_NOW - timedelta(days=1), start_time=None)]
        )

        assert len(_in_window(repo)) == 1

    def test_the_bound_may_be_given_in_any_zone(self, repo):
        """Both sides of the comparison must share an offset, and only one of
        them lives in the database. Measured on the live data: the same instant
        passed as a local-zone bound disagreed with the truth on 15 candidates,
        and as a UTC bound on none. The caller works in local time — the floor is
        local midnight — so canonicalising the bound is the repository's job.
        """
        eastern = timezone(timedelta(hours=-4))
        repo.save([_candidate(start_time=_NOW + timedelta(hours=1))])

        as_utc = repo.for_window(seen_since=_LOOKBACK, starting_after=_NOW)
        as_local = repo.for_window(
            seen_since=_LOOKBACK.astimezone(eastern),
            starting_after=_NOW.astimezone(eastern),
        )

        assert [c.id for c in as_local] == [c.id for c in as_utc] == ["c1"]

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
