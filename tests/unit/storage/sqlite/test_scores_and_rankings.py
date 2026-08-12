"""Behaviour only a real database has: the compound key and its cascades.

Deliberately not in the shared contract suite. An in-memory fake would have to
reimplement referential integrity to pass these, and a fake that enforces its
own rules is exactly the drift the contract exists to prevent.
"""

from __future__ import annotations

from datetime import date

import pytest
import sqlite3

from src.models.event_score import EventScore
from src.models.ranking import Ranking
from src.storage.sqlite.connection import connect, init_db
from src.storage.sqlite.rankings import SqliteRankingRepository
from src.storage.sqlite.scores import SqliteScoreRepository

RUN = date(2026, 8, 11)


@pytest.fixture
def db(tmp_path):
    path = tmp_path / "scores.db"
    init_db(path)
    conn = connect(path)
    try:
        conn.execute(
            "INSERT INTO events (id, source_type, created_at, updated_at) "
            "VALUES ('e1', 'test', '2026-08-11T00:00:00', '2026-08-11T00:00:00')"
        )
        conn.commit()
    finally:
        conn.close()
    return path


def _score(base_score: float = 0.5, **kwargs) -> EventScore:
    return EventScore(
        event_id="e1", run_date=RUN, base_score=base_score, match="yes", **kwargs
    )


def _ranking(**kwargs) -> Ranking:
    return Ranking(event_id="e1", run_date=RUN, final_score=0.6, rank=1, **kwargs)


def test_a_ranking_without_its_score_is_refused(db):
    """The FK is the guard that made the 2026-08-10 bug findable at all."""
    rankings = SqliteRankingRepository(db)

    with pytest.raises(sqlite3.IntegrityError):
        rankings.save([_ranking()])


def test_a_score_for_an_unknown_event_is_refused(db):
    scores = SqliteScoreRepository(db)

    with pytest.raises(sqlite3.IntegrityError):
        scores.save([EventScore(event_id="ghost", run_date=RUN, base_score=0.1, match="no")])


def test_replacing_a_runs_scores_cascades_its_rankings_away(db):
    """`rankings` and `score_reasons` both hang off `event_scores`."""
    scores, rankings = SqliteScoreRepository(db), SqliteRankingRepository(db)
    scores.save([_score()])
    rankings.save([_ranking()])

    scores.save([_score(base_score=0.9)])

    assert rankings.for_run(RUN) == []


def test_purging_an_event_takes_its_score_and_ranking_with_it(db):
    scores, rankings = SqliteScoreRepository(db), SqliteRankingRepository(db)
    scores.save([_score()])
    rankings.save([_ranking()])

    conn = connect(db)
    try:
        conn.execute("DELETE FROM events WHERE id = 'e1'")
        conn.commit()
    finally:
        conn.close()

    assert scores.for_run(RUN) == []
    assert rankings.for_run(RUN) == []
