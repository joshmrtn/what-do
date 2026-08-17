"""Tests for the shared pipeline: the scope predicate and the terminal step.

The terminal step exists so there is **one** call site for ranking. Two roots
each calling `RankingEngine.rank` themselves would have to agree about the scope
filter, the argument order and the save order forever, with nothing checking
that they still do.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from typing import Any, Callable
from zoneinfo import ZoneInfo

from src.composition.pipeline import finish_run, scope_filter, scope_floor
from src.config import (
    AppConfig,
    LocationConfig,
    ScoringConfig,
    ScrapingConfig,
    VenueDiscoveryConfig,
    WeatherConfig,
)
from src.models.event import Event
from src.models.tag import Tag
from src.scoring.ranking import RankingEngine
from src.scoring.similarity import Reason, SimilarityResult
from src.storage.memory.rankings import InMemoryRankingRepository
from src.storage.memory.scores import InMemoryScoreRepository

ZONE = ZoneInfo("America/New_York")
RUN_DATE = date(2025, 6, 21)
NOW = datetime(2025, 6, 21, 2, 0, tzinfo=ZONE)


def _config() -> AppConfig:
    return AppConfig(
        location=LocationConfig(42.52, -70.89, "01970", 10.0, "America/New_York"),
        scraping=ScrapingConfig(),
        venue_discovery=VenueDiscoveryConfig(blocklist_name_match_threshold=0.80),
        scoring=ScoringConfig(
            match_multiplier_yes=1.5,
            match_multiplier_maybe=1.0,
            match_multiplier_no=0.5,
            min_tags_per_event=5,
        ),
        weather=WeatherConfig(),
    )


def _event(event_id: str, start: datetime | None) -> Event:
    return Event(
        event_id=event_id,
        source_event_candidates=[],
        source_type="instagram",
        created_at=NOW,
        updated_at=NOW,
        title="Test Event",
        start_time=start,
        venue="The Jazz Cellar",
        tags=[Tag(text=f"tag{i}") for i in range(5)],
        setting="indoor",
        similarity=SimilarityResult(
            tag_score=0.4,
            summary_score=0.0,
            base_score=0.4,
            match="maybe",
            reasons=[
                Reason(
                    factor="like_similarity",
                    matched_preference="live music",
                    similarity=0.8,
                    contribution=0.4,
                    direction="positive",
                    tag="jazz",
                )
            ],
        ),
    )


def _passthrough(name: str, fn: Callable[[], Any], default: Any = None) -> Any:
    """The trivial driver: run the step, let anything it raises escape."""
    return fn()


class _OrderRecorder:
    """Records that `save` was called, then forwards it unchanged.

    Records and forwards, so it makes no behavioural claim of its own and cannot
    drift from the repository it wraps. Everything else passes straight through
    by `__getattr__`, so a repository growing a method does not break it.
    """

    def __init__(self, calls: list[str], name: str, inner: Any) -> None:
        self._calls = calls
        self._name = name
        self._inner = inner

    def __getattr__(self, attr: str) -> Any:
        inner = getattr(self._inner, attr)
        if attr != "save":
            return inner

        def recorded(*args: Any, **kwargs: Any) -> Any:
            self._calls.append(self._name)
            return inner(*args, **kwargs)

        return recorded


def _finish(
    events: list[Event],
    *,
    persist: bool = True,
    run_stage: Callable[..., Any] = _passthrough,
    scores_repo: Any = None,
    rankings_repo: Any = None,
    now: datetime = NOW,
) -> Any:
    return finish_run(
        events=events,
        run_date=RUN_DATE,
        now=now,
        config=_config(),
        ranking_engine=RankingEngine(_config()),
        score_repository=scores_repo or InMemoryScoreRepository(),
        ranking_repository=rankings_repo or InMemoryRankingRepository(),
        persist=persist,
        run_stage=run_stage,
    )


def test_scores_are_saved_before_rankings():
    """A ranking references its score by `(event_id, run_date)`.

    The foreign key refuses the write in the other order, so this is not a
    stylistic preference — it is the reason the terminal step is one function
    rather than two calls each root makes for itself.
    """
    calls: list[str] = []
    scores = _OrderRecorder(calls, "scores", InMemoryScoreRepository())
    rankings = _OrderRecorder(calls, "rankings", InMemoryRankingRepository())

    _finish(
        [_event("evt-1", datetime(2025, 6, 21, 20, 0, tzinfo=ZONE))],
        scores_repo=scores,
        rankings_repo=rankings,
    )

    assert calls == ["scores", "rankings"]


def test_an_event_outside_the_scope_is_not_ranked():
    """The scope filter runs inside the terminal step, not beside it.

    An event that finished before the run date's local midnight is not worth
    ranking, and a caller that forgot to filter would rank it.
    """
    outcome = _finish(
        [
            _event("keep", datetime(2025, 6, 21, 20, 0, tzinfo=ZONE)),
            _event("drop", datetime(2025, 6, 19, 20, 0, tzinfo=ZONE)),
        ]
    )

    assert [r.event_id for r in outcome.rankings] == ["keep"]
    assert outcome.rankable == 1


def test_nothing_is_persisted_when_persist_is_false():
    """`--dry-run` runs every stage for real and writes nothing.

    The read path relies on this same seam, so it is pinned here rather than
    only in the scheduler's own tests.
    """
    scores = InMemoryScoreRepository()
    rankings = InMemoryRankingRepository()

    outcome = _finish(
        [_event("evt-1", datetime(2025, 6, 21, 20, 0, tzinfo=ZONE))],
        persist=False,
        scores_repo=scores,
        rankings_repo=rankings,
    )

    assert outcome.rankings, "the ranking is still computed"
    assert scores.for_run(RUN_DATE) == []
    assert rankings.for_run(RUN_DATE) == []


def test_a_failed_ranking_persists_nothing():
    """When ranking fails, its default must not be written as an empty run.

    Saving `([], [])` would replace a good ranking with nothing, because both
    repositories replace by run date.
    """
    calls: list[str] = []
    scores = _OrderRecorder(calls, "scores", InMemoryScoreRepository())
    rankings = _OrderRecorder(calls, "rankings", InMemoryRankingRepository())

    def failing(name: str, fn: Callable[[], Any], default: Any = None) -> Any:
        if name == "ranking":
            return default
        return fn()

    outcome = _finish(
        [_event("evt-1", datetime(2025, 6, 21, 20, 0, tzinfo=ZONE))],
        run_stage=failing,
        scores_repo=scores,
        rankings_repo=rankings,
    )

    assert outcome.scores == []
    assert calls == []


def test_the_scope_floor_is_local_midnight_of_the_run_date():
    """Local, not UTC: an 11pm event is tomorrow in UTC.

    A floor derived from the UTC date would drop exactly the evening events the
    ranking exists to order.
    """
    floor = scope_floor(_config(), RUN_DATE)

    assert floor == datetime(2025, 6, 21, 0, 0, tzinfo=ZoneInfo("America/New_York"))


def test_an_undated_event_is_scoped_on_its_discovery_age():
    """Undated events are kept on discovery age, not dropped.

    The CLI has a labelled place for them, and dropping one would lose a real
    event to a failed extraction.
    """
    config = _config()
    in_scope = scope_filter(config, RUN_DATE, NOW)

    fresh = _event("fresh", None)
    stale = _event("stale", None)
    stale.created_at = NOW - timedelta(days=config.scraping.lookback_days + 1)

    assert in_scope(fresh) is True
    assert in_scope(stale) is False


def test_the_scope_filter_does_not_move_during_the_day():
    """The floor is the run date's midnight, never `now`.

    A read-time rescore re-runs this predicate hours after the batch did. If the
    floor tracked the clock, the rescore would silently drop events the batch
    ranked — and the listing would shrink for a reason nothing recorded.
    """
    config = _config()
    event = _event("afternoon", datetime(2025, 6, 21, 15, 0, tzinfo=ZONE))

    at_two_am = scope_filter(config, RUN_DATE, NOW)
    at_eight_pm = scope_filter(config, RUN_DATE, datetime(2025, 6, 21, 20, 0, tzinfo=ZONE))

    assert at_two_am(event) is True
    assert at_eight_pm(event) is True


def test_events_beyond_the_horizon_are_out_of_scope():
    """Ranking everything ever stored grows without bound."""
    config = _config()
    in_scope = scope_filter(config, RUN_DATE, NOW)
    beyond = _event(
        "beyond",
        datetime(2025, 6, 21, 20, 0, tzinfo=ZONE)
        + timedelta(days=config.scraping.horizon_days + 1),
    )

    assert in_scope(beyond) is False


def test_utc_timestamps_are_read_in_the_configured_zone():
    """An 11pm local event carries tomorrow's UTC date.

    Comparing the horizon against that date would misfile exactly the evening
    events this ranks.
    """
    config = _config()
    in_scope = scope_filter(config, RUN_DATE, NOW)
    late = _event("late", datetime(2025, 6, 22, 3, 0, tzinfo=timezone.utc))

    assert in_scope(late) is True
