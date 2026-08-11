"""Contract every ScoreRepository and RankingRepository implementation must satisfy.

Run against both the SQLite repositories and the in-memory ones, for the reasons
in `test_event_repository.py`.

Behaviour only a database can have — the compound foreign key from `rankings`
onto `event_scores`, and its cascade — is **not** here. It lives in
`test_sqlite_score_and_ranking.py`, because a fake reimplementing referential
integrity is the drift problem wearing a different hat.
"""

from __future__ import annotations

from datetime import date

import pytest

from src.models.event_score import EventScore
from src.models.ranking import Ranking
from src.scoring.similarity import Reason
from src.storage.db import connect, init_db
from src.storage.memory.rankings import InMemoryRankingRepository
from src.storage.memory.scores import InMemoryScoreRepository
from src.storage.sqlite.rankings import SqliteRankingRepository
from src.storage.sqlite.scores import SqliteScoreRepository

RUN = date(2026, 8, 11)
EARLIER = date(2026, 8, 10)


#: Every event id this suite scores. SQLite enforces the foreign key onto
#: `events`, so the rows have to exist before anything can reference them.
#: That is fixture setup, not a contract assertion — the key itself is tested
#: in `test_sqlite_score_and_ranking.py`, where a real database can enforce it.
_EVENT_IDS = ("e1", "a", "b", "c", "in-scope", "out-of-scope")


@pytest.fixture(params=["sqlite", "memory"])
def repos(request, tmp_path):
    """A (scores, rankings) pair per implementation, so each test runs twice."""
    if request.param == "sqlite":
        path = tmp_path / "scores.db"
        init_db(path)
        conn = connect(path)
        try:
            conn.executemany(
                "INSERT INTO events (id, source_type, created_at, updated_at) "
                "VALUES (?, 'test', '2026-08-11T00:00:00', '2026-08-11T00:00:00')",
                [(event_id,) for event_id in _EVENT_IDS],
            )
            conn.commit()
        finally:
            conn.close()
        return SqliteScoreRepository(path), SqliteRankingRepository(path)
    return InMemoryScoreRepository(), InMemoryRankingRepository()


def _reason(factor: str = "like_similarity", tag: str | None = "karaoke") -> Reason:
    return Reason(
        factor=factor,
        matched_preference="karaoke night",
        similarity=0.87,
        contribution=0.8,
        direction="positive",
        tag=tag,
    )


def _score(event_id: str = "e1", run_date: date = RUN, **kwargs) -> EventScore:
    defaults = dict(
        event_id=event_id,
        run_date=run_date,
        tag_score=0.51,
        summary_score=0.33,
        base_score=0.61,
        tag_confidence=0.8,
        match="yes",
        reasons=[_reason()],
    )
    defaults.update(kwargs)
    return EventScore(**defaults)


def _ranking(event_id: str = "e1", run_date: date = RUN, rank: int = 1, **kwargs) -> Ranking:
    defaults = dict(
        event_id=event_id,
        run_date=run_date,
        weather_adjustment=0.05,
        final_score=0.66,
        rank=rank,
    )
    defaults.update(kwargs)
    return Ranking(**defaults)


class TestScoreRoundTrip:
    def test_every_field_survives(self, repos):
        scores, _ = repos
        scores.save([_score()])

        stored = scores.for_run(RUN)[0]

        assert stored.event_id == "e1"
        assert stored.run_date == RUN
        assert stored.tag_score == pytest.approx(0.51)
        assert stored.summary_score == pytest.approx(0.33)
        assert stored.base_score == pytest.approx(0.61)
        assert stored.tag_confidence == pytest.approx(0.8)
        assert stored.match == "yes"

    def test_the_components_behind_base_score_are_not_dropped(self, repos):
        """They were written as hardcoded NULL until 2026-08-11, all 861 rows."""
        scores, _ = repos
        scores.save([_score(tag_score=0.42, summary_score=0.17)])

        stored = scores.for_run(RUN)[0]

        assert (stored.tag_score, stored.summary_score) == (
            pytest.approx(0.42),
            pytest.approx(0.17),
        )

    def test_reasons_load_as_structured_objects_in_order(self, repos):
        scores, _ = repos
        ordered = [_reason(tag="first"), _reason(tag="second"), _reason(tag="third")]
        scores.save([_score(reasons=ordered)])

        stored = scores.for_run(RUN)[0]

        assert [r.tag for r in stored.reasons] == ["first", "second", "third"]
        assert stored.reasons[0].matched_preference == "karaoke night"

    def test_a_score_with_no_reasons_round_trips_empty(self, repos):
        scores, _ = repos
        scores.save([_score(reasons=[])])

        assert scores.for_run(RUN)[0].reasons == []


class TestScoreReplacement:
    def test_rerunning_a_date_replaces_that_runs_scores(self, repos):
        scores, _ = repos
        scores.save([_score(base_score=0.1)])
        scores.save([_score(base_score=0.9)])

        stored = scores.for_run(RUN)
        assert len(stored) == 1
        assert stored[0].base_score == pytest.approx(0.9)

    def test_replacement_does_not_touch_another_night(self, repos):
        scores, _ = repos
        scores.save([_score(run_date=EARLIER, base_score=0.1)])
        scores.save([_score(run_date=RUN, base_score=0.9)])

        assert scores.for_run(EARLIER)[0].base_score == pytest.approx(0.1)

    def test_saving_nothing_does_not_clear_a_stored_run(self, repos):
        scores, _ = repos
        scores.save([_score()])
        scores.save([])

        assert len(scores.for_run(RUN)) == 1

    def test_an_unknown_run_reads_back_empty(self, repos):
        scores, _ = repos

        assert scores.for_run(RUN) == []


class TestRankings:
    def test_every_field_survives(self, repos):
        scores, rankings = repos
        scores.save([_score()])
        rankings.save([_ranking()])

        stored = rankings.for_run(RUN)[0]

        assert stored.event_id == "e1"
        assert stored.run_date == RUN
        assert stored.weather_adjustment == pytest.approx(0.05)
        assert stored.final_score == pytest.approx(0.66)
        assert stored.rank == 1

    def test_placements_read_back_in_rank_order(self, repos):
        scores, rankings = repos
        scores.save([_score("a"), _score("b"), _score("c")])
        rankings.save(
            [_ranking("c", rank=3), _ranking("a", rank=1), _ranking("b", rank=2)]
        )

        assert [r.event_id for r in rankings.for_run(RUN)] == ["a", "b", "c"]

    def test_rerunning_a_date_replaces_that_runs_placements(self, repos):
        scores, rankings = repos
        scores.save([_score()])
        rankings.save([_ranking(rank=5)])
        rankings.save([_ranking(rank=1)])

        stored = rankings.for_run(RUN)
        assert len(stored) == 1
        assert stored[0].rank == 1

    def test_saving_nothing_does_not_clear_a_stored_run(self, repos):
        scores, rankings = repos
        scores.save([_score()])
        rankings.save([_ranking()])
        rankings.save([])

        assert len(rankings.for_run(RUN)) == 1


class TestScoresOutliveRankings:
    """The point of the split: an event can be scored and not ranked.

    ~258 events a run are scored and thrown away today because the API had no
    way to say "keep the verdict, skip the placement".
    """

    def test_a_score_can_exist_with_no_ranking(self, repos):
        scores, rankings = repos
        scores.save([_score("in-scope"), _score("out-of-scope")])
        rankings.save([_ranking("in-scope")])

        assert {s.event_id for s in scores.for_run(RUN)} == {"in-scope", "out-of-scope"}
        assert [r.event_id for r in rankings.for_run(RUN)] == ["in-scope"]


class TestLatestRunDate:
    def test_nothing_ranked_yet_reads_back_as_nothing(self, repos):
        _, rankings = repos

        assert rankings.latest_run_date() is None

    def test_the_most_recent_ranked_night_wins(self, repos):
        scores, rankings = repos
        scores.save([_score(run_date=EARLIER), _score(run_date=RUN)])
        rankings.save([_ranking(run_date=EARLIER), _ranking(run_date=RUN)])

        assert rankings.latest_run_date() == RUN

    def test_a_scored_but_unranked_night_does_not_count(self, repos):
        """A run that died before ranking has nothing the CLI can display."""
        scores, rankings = repos
        scores.save([_score(run_date=EARLIER), _score(run_date=RUN)])
        rankings.save([_ranking(run_date=EARLIER)])

        assert rankings.latest_run_date() == EARLIER
