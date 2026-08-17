"""The pipeline both roots share: the scope predicate and the terminal step.

Two entry points need to turn scored events into a stored ordering — the
overnight batch, and the read-time rescore behind `what-do`. Ranking is the one
place where "they must call it the same way" is not good enough: the scope
filter, the argument order and the save order all have to agree, and nothing
would check that they still did.

So there is **one call site**, here, which both roots depend on and neither
owns. What stays with each root is the part that genuinely differs: whether to
persist at all, and what to do when a step fails.

The save order is not policy. A `Ranking` references its `EventScore` by
`(event_id, run_date)` and the foreign key refuses the write otherwise, so the
order is a fact about the schema and belongs with the thing that writes it. The
scope filter is not policy either — it is the definition of "rankable".
"""

from __future__ import annotations

import json
import zoneinfo
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Any, Callable

from src.config import AppConfig
from src.enrichment.weather import latest_forecast
from src.models.event import Event
from src.models.event_score import EventScore
from src.models.ranking import Ranking
from src.scoring.ranking import RankingEngine
from src.storage.protocols import RankingRepository, ScoreRepository

#: How a root runs one step: a name, the work, and what to fall back to.
#:
#: The batch passes a wrapper that records the failure and carries on with a
#: partial outcome; the read path passes one that abandons the rescore and
#: leaves the stored ranking alone. Both are policies about failure, which is
#: why neither lives here.
StageRunner = Callable[..., Any]


class RescoreUnavailable(RuntimeError):
    """Raised when a read-time rescore cannot produce an honest ordering.

    The failure that makes this necessary: `EmbeddingStage` treats a failed
    embedding as *per-event, non-fatal* — it marks the event and moves on,
    because one bad event must not cost a batch the other nine hundred. On the
    read path that degradation is exactly wrong. The refusing provider fires,
    the event scores zero on the semantic half, and the rescore writes that
    zero over a perfectly good stored score.

    So the seal is not the refusing provider alone. It is the refusing provider
    *plus* this: if anything on this path needed a vector, the whole rescore is
    abandoned and the stored ranking stands.
    """


def default_stage_runner(name: str, fn: Callable[[], Any], default: Any = None) -> Any:
    """Run a step and let anything it raises escape.

    The honest default for a caller with no failure policy of its own. A root
    that needs one passes it in rather than inheriting a silent shrug.
    """
    return fn()


@dataclass(frozen=True)
class RankingOutcome:
    """What the terminal step produced.

    Attributes:
        scores: One verdict per ranked event, in placement order.
        rankings: Each ranked event's placement, best first.
        rankable: How many events passed the scope filter. Reported separately
            because it is the count a run summary wants, and it is not
            recoverable from `rankings` once ranking has failed.
    """

    scores: list[EventScore] = field(default_factory=list)
    rankings: list[Ranking] = field(default_factory=list)
    rankable: int = 0


def load_blocklist(path: Path) -> list[str]:
    """Read `blocklist.json`, treating an absent file as an empty blocklist.

    Here rather than in either root because both rank now, and the blocklist has
    exactly one source of truth. Two readers is how the ranking a rescore
    produces comes to differ from the one the batch produced, for a reason
    nothing records.

    A user who never wrote one is a normal deployment, not a failure.
    """
    if not path.exists():
        return []
    return [str(entry) for entry in json.loads(path.read_text())]


def scope_floor(config: AppConfig, run_date: date) -> datetime:
    """The instant before which an event is over: local midnight of the run date.

    Shared by `scope_filter` and the candidate window, and shared **only** here.
    The two ask different questions — one *"is this worth ranking?"*, the other
    *"is this record still live?"* — but a finished event is both unrankable and
    stale, so this one fact answers both, and writing it twice is how the two
    drift apart.

    Their *ceilings* are deliberately not shared: a horizon expresses how far
    ahead we care to look, which says nothing about whether a record is current,
    and `for_window` is the only path a stored candidate has back into a batch.

    Local, not UTC: an event at 11pm local is tomorrow in UTC, and a floor
    derived from that would drop exactly the evening events we rank.
    """
    tz = zoneinfo.ZoneInfo(config.location.timezone)
    return datetime.combine(run_date, time.min, tzinfo=tz)


def scope_filter(
    config: AppConfig, run_date: date, now: datetime
) -> Callable[[Event], bool]:
    """Build the predicate for whether an event is worth ranking this run.

    Ranking everything ever stored grows without bound; ranking only tonight
    discards the lookahead the calendar feeds exist for. Undated events are kept
    on discovery age instead, because the CLI has a labelled section for them
    and dropping one would lose a real event to a failed extraction.

    Note that `now` reaches only the undated arm. A dated event is judged
    against the run date's local midnight, so the predicate does not move as the
    evening wears on — which is what lets a read-time rescore re-rank the same
    population the batch did, hours later, without silently shrinking it.
    """
    horizon = run_date + timedelta(days=config.scraping.horizon_days)
    lookback_cutoff = now - timedelta(days=config.scraping.lookback_days)
    tz = zoneinfo.ZoneInfo(config.location.timezone)
    floor = scope_floor(config, run_date)

    def in_scope(event: Event) -> bool:
        if event.start_time is None:
            return event.created_at >= lookback_cutoff
        if event.start_time < floor:
            return False
        # Local date, not UTC: an event at 11pm local is tomorrow in UTC, and
        # filtering on that would misfile exactly the evening events we rank.
        return event.start_time.astimezone(tz).date() <= horizon

    return in_scope


def finish_run(
    *,
    events: list[Event],
    run_date: date,
    now: datetime,
    config: AppConfig,
    ranking_engine: RankingEngine,
    score_repository: ScoreRepository,
    ranking_repository: RankingRepository,
    persist: bool,
    run_stage: StageRunner = default_stage_runner,
    before_persist: Callable[[], None] | None = None,
) -> RankingOutcome:
    """Scope, rank and store one run's ordering.

    The terminal step of the pipeline, and the only place `RankingEngine.rank`
    is called. A guard in `tests/unit/test_composition_is_the_only_namer.py`
    keeps it that way.

    Args:
        events: Scored events, each carrying a `similarity` result.
        run_date: The run whose ordering this is. For a read-time rescore this
            is the **loaded run's** date, never today's: the two frames disagree
            either side of midnight, and deriving a fresh one mints a run no
            batch produced.
        now: Injected clock, reaching only the undated arm of the scope filter.
        config: Supplies the horizon, the lookback and the zone.
        ranking_engine: The engine to rank with.
        score_repository: Where verdicts are written.
        ranking_repository: Where placements are written.
        persist: False runs everything and writes nothing, which is what
            `--dry-run` does and what the read path reuses.
        run_stage: How to run each step and what to do when it fails.
        before_persist: Run after a successful rank and before the first
            write. It exists for one reason: `INSERT OR REPLACE INTO events`
            is a DELETE and an INSERT, so `event_scores` cascades away and
            takes `rankings` with it. Saving an event therefore destroys its
            placement, and doing that *before* knowing the new one exists
            leaves a run with no ranking at all.

    Returns:
        The scores, the placements, and how many events were in scope.
    """
    in_scope = scope_filter(config, run_date, now)
    rankable = [event for event in events if in_scope(event)]

    scores, rankings = run_stage(
        "ranking", lambda: ranking_engine.rank(rankable, run_date), ([], [])
    )

    # Guarded on `scores` rather than on `rankings`: both repositories replace
    # by run date, so writing a failed ranking's empty default would delete a
    # good stored ordering and leave nothing in its place.
    if persist and scores:
        if before_persist is not None:
            before_persist()
        run_stage("save_scores", lambda: score_repository.save(scores), None)
        run_stage("save_rankings", lambda: ranking_repository.save(rankings), None)

    return RankingOutcome(scores=scores, rankings=rankings, rankable=len(rankable))


def run_view_tail(
    *,
    events: list[Event],
    run_date: date,
    now: datetime,
    config: AppConfig,
    enrichment_service: Any,
    embedding_stage: Any,
    similarity_stage: Any,
    ranking_engine: RankingEngine,
    event_repository: Any,
    score_repository: ScoreRepository,
    ranking_repository: RankingRepository,
    forecast_fresh_since: datetime,
    run_stage: StageRunner = default_stage_runner,
) -> RankingOutcome:
    """Re-run the pipeline from enrichment, for a caller that cannot extract.

    The batch's tail with two stages deliberately absent:

    * **Extraction** costs minutes an event, which is the whole reason the read
      path exists as a separate thing. It is absent structurally rather than
      skipped by a flag — there is no stage here to reach.
    * **Semantic dedup** is absent because its inputs cannot have changed. It
      compares summary and tag vectors, and those move only when extraction
      moves them, so re-running it could reach no conclusion it has not already
      reached — while writing supersessions and decision rows from a read. All
      risk, no possible difference.

    Synthetic activities are regenerated rather than carried, because their
    rules are *conditioned on the weather* — a refreshed forecast can make an
    evening walk qualify or stop qualifying, which is the clearest thing a real
    re-run gives that patching a number never could. Their ids are derived from
    the rule and the run date, so regeneration overwrites rather than
    duplicating, and the stored ones are dropped first so one run cannot hold
    two rows for the same activity.

    Args:
        events: The run's ranked events, as stored.
        run_date: The run being recomputed — the loaded run's own date.
        now: Injected clock.
        config: Horizon, lookback and zone for the scope filter.
        enrichment_service: Refreshes weather and regenerates synthetics.
        embedding_stage: Runs, and is a no-op while tags are unchanged.
        similarity_stage: Re-scores against the loaded preferences.
        ranking_engine: The terminal engine.
        event_repository: Where refreshed events are written back. Needed
            because a regenerated synthetic has no row until it is saved, and a
            placement referencing an absent event violates the foreign key.
        score_repository: Where the new verdicts go.
        ranking_repository: Where the new placements go.
        forecast_fresh_since: The oldest forecast this rescore may be built
            on. Enrichment swallows a failed fetch and returns no weather,
            so without this the rescore would happily write an ordering
            computed with the weather term missing entirely.
        run_stage: How to run each step and what to do when it fails.

    Returns:
        The recomputed scores and placements.
    """
    # Dropped rather than carried: enrichment appends a freshly generated set,
    # and keeping the stored ones would put two rows with one id into a single
    # ranking.
    carried = [event for event in events if not event.is_synthetic]

    enriched = run_stage(
        "enrichment", lambda: enrichment_service.enrich(carried, run_date), carried
    )

    # A failed forecast fetch is *logged and swallowed* by enrichment, which is
    # right for a batch — one bad night must not stop a run — and wrong here.
    # The rescore exists to apply a fresher forecast; without one it would write
    # back an ordering with the weather term silently absent, which is strictly
    # worse than the stored ranking it replaces.
    issued = latest_forecast(enriched)
    if issued is None or issued < forecast_fresh_since:
        raise RescoreUnavailable(
            "no fresh forecast: the rescore would score with weather missing"
        )
    embedded = run_stage("embedding", lambda: embedding_stage.process(enriched), enriched)

    # Not a formality. The stage degrades per event by design, so without this
    # an event whose vectors are missing scores zero on the semantic half and
    # that zero is written over a good stored score — the rescore actively
    # making the listing worse, silently.
    unembeddable = [
        event.event_id
        for event in embedded
        if event.metadata.get("embedding_failed") or event.metadata.get("embedding_skipped")
    ]
    if unembeddable:
        raise RescoreUnavailable(
            f"{len(unembeddable)} event(s) need embedding, which is a model call: "
            f"{', '.join(sorted(unembeddable)[:3])}"
        )

    scored = run_stage("similarity", lambda: similarity_stage.process(embedded), embedded)

    return finish_run(
        events=scored,
        run_date=run_date,
        now=now,
        config=config,
        ranking_engine=ranking_engine,
        score_repository=score_repository,
        ranking_repository=ranking_repository,
        persist=True,
        run_stage=run_stage,
        # After the rank, never before. A regenerated synthetic has no row
        # until this runs and its placement's foreign key needs one, but
        # saving an event cascades its stored placement away — so this is
        # the one window where both are true.
        before_persist=lambda: event_repository.save(scored),
    )
