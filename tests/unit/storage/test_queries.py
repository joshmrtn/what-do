"""The cross-aggregate read, driven entirely by in-memory repositories.

No SQLite anywhere: that is the payoff of having a second implementation, and
the reason the join lives outside the repositories at all.
"""

from __future__ import annotations

from datetime import date, datetime, timezone

from src.models.event import Event
from src.models.event_score import EventScore
from src.models.ranking import Ranking
from src.storage.memory.events import InMemoryEventRepository
from src.storage.memory.rankings import InMemoryRankingRepository
from src.storage.memory.scores import InMemoryScoreRepository
from src.storage.queries import load_ranked_events

RUN = date(2026, 8, 11)
EARLIER = date(2026, 8, 10)
_NOW = datetime(2026, 8, 11, tzinfo=timezone.utc)


def _repos(events=(), scores=(), rankings=()):
    event_repo = InMemoryEventRepository()
    event_repo.save(list(events))
    score_repo = InMemoryScoreRepository()
    score_repo.save(list(scores))
    ranking_repo = InMemoryRankingRepository()
    ranking_repo.save(list(rankings))
    return event_repo, score_repo, ranking_repo


def _event(event_id: str) -> Event:
    return Event(
        event_id=event_id,
        source_event_candidates=[],
        source_type="test",
        created_at=_NOW,
        updated_at=_NOW,
        title=f"Event {event_id}",
    )


def _score(event_id: str, run_date: date = RUN) -> EventScore:
    return EventScore(event_id=event_id, run_date=run_date, base_score=0.5, match="yes")


def _ranking(event_id: str, rank: int, run_date: date = RUN) -> Ranking:
    return Ranking(event_id=event_id, run_date=run_date, final_score=0.6, rank=rank)


def test_nothing_ranked_yet_reads_back_empty():
    assert load_ranked_events(*_repos()) == []


def test_each_placement_is_paired_with_its_event_and_score():
    repos = _repos([_event("a")], [_score("a")], [_ranking("a", 1)])

    view = load_ranked_events(*repos)

    assert len(view) == 1
    assert view[0].event.event_id == "a"
    assert view[0].score.match == "yes"
    assert view[0].ranking.rank == 1


def test_the_batchs_order_is_preserved():
    repos = _repos(
        [_event("a"), _event("b"), _event("c")],
        [_score("a"), _score("b"), _score("c")],
        [_ranking("c", 1), _ranking("a", 2), _ranking("b", 3)],
    )

    assert [r.event.event_id for r in load_ranked_events(*repos)] == ["c", "a", "b"]


def test_it_defaults_to_the_latest_ranked_night():
    repos = _repos(
        [_event("old"), _event("new")],
        [_score("old", EARLIER), _score("new", RUN)],
        [_ranking("old", 1, EARLIER), _ranking("new", 1, RUN)],
    )

    assert [r.event.event_id for r in load_ranked_events(*repos)] == ["new"]


def test_an_earlier_night_can_be_pinned():
    repos = _repos(
        [_event("old"), _event("new")],
        [_score("old", EARLIER), _score("new", RUN)],
        [_ranking("old", 1, EARLIER), _ranking("new", 1, RUN)],
    )

    view = load_ranked_events(*repos, run_date=EARLIER)

    assert [r.event.event_id for r in view] == ["old"]


def test_a_placement_whose_event_is_gone_is_skipped_not_fatal():
    """One purged event must not take down the whole listing."""
    repos = _repos([_event("kept")], [_score("kept"), _score("gone")],
                   [_ranking("kept", 1), _ranking("gone", 2)])

    assert [r.event.event_id for r in load_ranked_events(*repos)] == ["kept"]


def test_a_scored_event_with_no_placement_is_not_in_the_view():
    """Out-of-scope events keep their score and never reach the screen."""
    repos = _repos(
        [_event("in"), _event("out")],
        [_score("in"), _score("out")],
        [_ranking("in", 1)],
    )

    assert [r.event.event_id for r in load_ranked_events(*repos)] == ["in"]
