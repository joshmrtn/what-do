"""The overnight batch orchestrator.

A composition root and a sequencer. It owns wiring and failure policy; it owns
no domain logic. Every rule lives in the stage that owns it.

The save points are the point of the design. Extraction costs roughly three
minutes an event locally, so a batch that only persisted at the end would throw
that away on any crash and pay it again every night. Saving after extraction and
after embedding activates the skip-if-done branches both stages already carry,
which is what makes a re-run incremental and a crash survivable.
"""

from __future__ import annotations

import argparse
import json
import sys
import zoneinfo
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, TextIO

from src.composition.batch import BatchDependencies, build_dependencies
from src.composition.pipeline import finish_run, scope_filter, scope_floor
import yaml

from src.config import (
    DEFAULT_BLOCKLIST_PATH,
    DEFAULT_CONFIG_PATH,
    DEFAULT_DISLIKES_PATH,
    DEFAULT_LIKES_PATH,
    AppConfig,
    load_config,
)
from src.enrichment.service import EnrichmentService
from src.ingestion.id_churn import ChurnTally
from src.ingestion.latch import LatchReport, arm_latches
from src.storage.identity_state import IdentityStateStore
from src.storage.rekey import rekey_to_content_ids
from src.storage.snapshot import snapshot_database
from src.ingestion.ingestion_service import IngestionService, SourceTally
from src.models.event import Event
from src.observability.heartbeat import HEARTBEAT_PATH, HeartbeatFile
from src.observability.reporter import Progress, ProgressFn, ProgressLog
from src.models.event_candidate import EventCandidate
from src.models.event_score import EventScore
from src.models.preference_revision import PreferenceRevision
from src.models.ranking import Ranking
from src.normalization.reconcile import reconcile
from src.normalization.semantic_dedup import SemanticDeduplicationEngine
from src.normalization.service import NormalizationService
from src.processing.extraction_stage import ExtractionStage
from src.scoring.embedding_stage import EmbeddingStage
from src.scoring.ranking import RankingEngine
from src.scoring.refit import run_refit
from src.scoring.similarity_stage import SimilarityStage
from src.config_check import Finding as ConfigFinding, check_config_file
from src.storage.schema_check import Finding, check_database, format_findings
from src.storage.sqlite.connection import DEFAULT_DB_PATH, init_db
from src.normalization.decision_sampling import select_for_storage
from src.normalization.deduplicator import MergeDecision
from src.storage.extraction_observations import ExtractionObservation
from src.storage.protocols import (
    CurveStateRepository,
    ExtractionObservationRepository,
    CandidateRepository,
    DedupDecisionRepository,
    EventRepository,
    PreferenceRevisionRepository,
    RankingRepository,
    RunRepository,
    ScoreRepository,
)
from src.utils.llm_transcript import LLMTranscript
from src.utils.logging import StructuredLogger, get_logger

#: Where the batch wrapper already keeps its per-run logs, so a transcript
#: lands beside the log it belongs to.
DEFAULT_LOG_DIR = Path("logs")
DEFAULT_SEEDS_PATH = Path("data/seeds.yaml")


def _default_now() -> datetime:
    """The clock the batch runs on when nothing injects one.

    Timezone-aware deliberately. `datetime.now` returns a naive time, and
    sources that state their own offset — Do617, every ICS feed — are compared
    against this during ingestion; the mismatch raised `can't compare
    offset-naive and offset-aware datetimes` and ended the fetch.
    """
    return datetime.now(timezone.utc)


@dataclass
class BatchResult:
    """What one run of the batch did."""

    outcome: str
    stage_counts: dict[str, int] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    skipped_sources: list[str] = field(default_factory=list)
    scores: list[EventScore] = field(default_factory=list)
    rankings: list[Ranking] = field(default_factory=list)
    #: What each source fetched and kept. A total cannot say which of seventeen
    #: sources went quiet, and that is the question a failed run always asks.
    per_source: dict[str, SourceTally] = field(default_factory=dict)
    #: Sources that raised during the fetch, by name.
    failed_sources: list[str] = field(default_factory=list)
    #: Whether each source's own ids identified anything, by source_type. A
    #: source whose ids re-mint every night cannot be tracked across runs, and
    #: nothing else in the batch would notice.
    churn: dict[str, ChurnTally] = field(default_factory=dict)
    #: What the churn latch did, if anything. Rare and worth announcing: it
    #: changes how a source is identified, permanently.
    latch: LatchReport = field(default_factory=LatchReport)
    #: What the nightly refit decided, or None where it did not run. A gate that
    #: declines and a refit that never ran are the same silence otherwise, and
    #: the batch spent a night in the second state looking like the first.
    refit: str | None = None


def run_batch(
    *,
    config: AppConfig,
    db_path: Path,
    ingestion_service: IngestionService,
    normalization_service: NormalizationService,
    enrichment_service: EnrichmentService,
    extraction_stage: ExtractionStage,
    embedding_stage: EmbeddingStage,
    semantic_deduplicator: SemanticDeduplicationEngine,
    similarity_stage: SimilarityStage,
    ranking_engine: RankingEngine,
    logger: StructuredLogger,
    run_date: date,
    get_now: Callable[[], datetime] = _default_now,
    skipped_sources: list[str] | None = None,
    skip_ingest: bool = False,
    dry_run: bool = False,
    ingest_only: bool = False,
    raw_dump_fn: Callable[[list[Any]], None] | None = None,
    heartbeat_path: Path | None = None,
    candidate_repository: CandidateRepository,
    event_repository: EventRepository,
    run_repository: RunRepository,
    score_repository: ScoreRepository,
    ranking_repository: RankingRepository,
    dedup_decision_repository: DedupDecisionRepository,
    curve_state_repository: CurveStateRepository,
    extraction_observation_repository: ExtractionObservationRepository,
    preference_revision_repository: PreferenceRevisionRepository,
    identity_state_repository: IdentityStateStore,
    preference_revision: PreferenceRevision,
) -> BatchResult:
    """Run one overnight batch, from ingestion through persisted recommendations.

    Args:
        config: Loaded application config.
        db_path: Path to the SQLite database.
        ingestion_service: Fetches and persists candidates.
        normalization_service: Normalizes and runs dedup pass 1.
        enrichment_service: Weather, astronomy, synthetic activities, movies.
        extraction_stage: LLM Pass 1.
        embedding_stage: Tag and summary vectors.
        semantic_deduplicator: Dedup pass 2, over embeddings.
        similarity_stage: Attaches `event.similarity`.
        ranking_engine: Terminal step; returns recommendations.
        logger: Structured logger.
        run_date: The date this batch is for.
        get_now: Injectable clock.
        skipped_sources: Sources the composition root could not build, normally
            for a missing credential. Reported, never inferred here.
        skip_ingest: Re-run the pipeline over already-fetched candidates without
            touching the network.
        dry_run: Run every stage but persist no events or recommendations.
            Ingestion still writes candidates, since it owns that write itself.
        ingest_only: Fetch, filter and stop. Proves every source still works in
            minutes rather than the hours a full run costs, and leaves the
            candidates behind so `skip_ingest` can process them later.
        heartbeat_path: Where to write live progress for `what-do --status`,
            or None to run unwatched. Passed rather than defaulted: a default
            only production reaches is a default no test exercises, and every
            suite run would otherwise write over the one path a live batch uses.
        raw_dump_fn: Given every candidate as fetched, with the reason any was
            discarded. Supplied only when a diagnostic run asks for the dump,
            since collecting it holds the whole fetch in memory.
        candidate_repository: Where fetched candidates are read back from.
            Defaults to the SQLite repository over `db_path`, on the same terms
            as `event_repository`.
        event_repository: Where events are read and written. Defaults to the
            SQLite repository over `db_path`; injected for testing, which is
            what lets the pipeline run without a database at all.
        run_repository: Where the run's history row is written. Defaults to the
            SQLite repository over `db_path`, on the same terms as
            `event_repository`.
        score_repository: Where each event's verdict is written, same terms.
        ranking_repository: Where each in-scope event's placement is written,
            same terms.
        dedup_decision_repository: Where each dedup comparison is written —
            the merges and, more importantly, the rejections.
        preference_revision_repository: Where the preference snapshot is
            recorded, so a stored score can be attributed to what it was
            measured against.
        preference_revision: What the preference files said when the
            composition root loaded them.

    Returns:
        BatchResult describing the outcome, per-stage counts, and any errors.
    """
    events_repo = event_repository
    runs_repo = run_repository
    scores_repo = score_repository
    candidates_repo = candidate_repository
    rankings_repo = ranking_repository
    dedup_decisions_repo = dedup_decision_repository
    result = BatchResult(outcome="success", skipped_sources=list(skipped_sources or []))
    now = get_now()

    # Neither a dry run nor an ingest-only run did a batch, so recording one
    # would pollute the only durable record of what the nightly runs actually did.
    # The scoring constants ride with the run. `config.yaml` is gitignored,
    # so a score whose constants have since been tuned is otherwise
    # unexplainable — the number survives and the arithmetic does not.
    # Recorded before the run row, because the row references it. Keyed on
    # content, so an unedited preference file resolves to the revision it
    # already has rather than writing one every night.
    revision_id = (
        None
        if dry_run or ingest_only
        else preference_revision_repository.record(preference_revision)
    )
    run_id = (
        None
        if dry_run or ingest_only
        else runs_repo.start(
            now,
            scoring_config=_scoring_provenance(config),
            dedup_config=_dedup_provenance(config),
            preference_revision_id=revision_id,
        )
    )

    def _record_decisions(decisions: list[MergeDecision]) -> None:
        """Store what a dedup pass concluded, sampled down to what is worth keeping.

        Skipped entirely without a run id: a dry run persists nothing, and
        there is no `run_history` row for these to reference — the thresholds
        behind a verdict are recoverable only through it.
        """
        if run_id is None or not decisions:
            return
        dedup_decisions_repo.save(
            select_for_storage(
                decisions,
                floor=config.deduplication.decision_floor,
                denominator=config.deduplication.decision_sample_denominator,
            ),
            run_id=run_id,
            now=now,
        )

    def _finish() -> BatchResult:
        """Complete the history row and hand back the result.

        Deliberately not a `finally`: a run that dies outside the stage
        wrappers should leave its row with a `started_at` and no
        `completed_at`, because that is what a crash looks like. Marking it
        finished on the way out would erase the one signal the start-row
        exists to give.
        """
        # Before the history row is closed, not after: the pair is read
        # together, and a moment where the run is complete *and* a heartbeat
        # still names it is a moment `--status` would call a death.
        if heartbeat is not None:
            heartbeat.clear()
        if run_id is not None:
            runs_repo.finish(
                run_id,
                outcome=result.outcome,
                completed_at=get_now(),
                stage_counts=result.stage_counts,
                errors=result.errors,
                skipped_sources=result.skipped_sources,
            )
        return result

    def _stage(name: str, fn: Callable[[], Any], default: Any = None) -> Any:
        """Run one stage, recording failure rather than ending the batch.

        Per-item failures are already non-fatal inside the stages. This covers
        the stage-level case: whatever has been saved stays saved, and the batch
        carries on with what it has.
        """
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 — a stage must not end the batch
            message = f"{name} failed: {exc}"
            result.errors.append(message)
            result.outcome = "partial"
            logger.error(message, component="batch", duration_ms=0)
            return default

    def _save(events: list[Event]) -> None:
        if not dry_run and events:
            events_repo.save(events)

    # One file for the whole run, and only for a run that has a `run_id` to
    # own it: a dry run holds no lock and writes no history row, so a heartbeat
    # from one could not be told from a real batch that died.
    heartbeat = (
        HeartbeatFile(heartbeat_path, run_id=run_id)
        if heartbeat_path is not None and run_id is not None
        else None
    )

    def _progress(stage_name: str) -> ProgressFn:
        """Where one stage's reports go: the log, and the live state file.

        A policy per stage, because each keeps its own reckoning — sharing an
        instance would carry extraction's start time and milestone into
        embedding, which runs after it against a different queue.

        Two sinks with two cadences off one report. The log is rationed,
        because a person reads it; the file is written every time, because
        `--status` asks it a question the log cannot answer.
        """
        log = ProgressLog(
            logger,
            milestone_fraction=config.observability.progress_milestone_fraction,
            heartbeat=timedelta(minutes=config.observability.progress_heartbeat_minutes),
        )
        if heartbeat is None:
            return log

        def report(progress: Progress) -> None:
            log(progress)
            heartbeat(progress)

        return report

    def _save_one(event: Event) -> None:
        """Persist a single freshly extracted event.

        Extraction checkpoints after every event now that one save no longer
        costs a rewrite of the whole corpus, so this takes the event rather than
        the list it belongs to.
        """
        if not dry_run:
            events_repo.save_one(event)

    fetched: list[EventCandidate] = []
    if skip_ingest:
        logger.info("skipping ingestion", component="batch", duration_ms=0)
    else:
        ingested = _stage(
            "ingestion",
            lambda: ingestion_service.run(
                get_now=get_now,
                persist=not dry_run,
                collect_raw=raw_dump_fn is not None,
            ),
        )
        if ingested is not None:
            result.stage_counts["ingested"] = ingested.accepted
            result.per_source = ingested.per_source
            result.failed_sources = list(ingested.failed_sources)
            result.churn = ingested.churn
            # After ingestion, deliberately: this run's candidates are already
            # stored, so re-keying here covers them along with everything that
            # accumulated before. A dry run persisted nothing, so it has no
            # rows to re-key and no evidence it is entitled to bank.
            if not dry_run:
                result.latch = _arm_identity_latches(
                    ingested.churn,
                    state=identity_state_repository,
                    config=config,
                    db_path=db_path,
                    logger=logger,
                    get_now=get_now,
                )
            if raw_dump_fn is not None:
                raw_dump_fn(ingested.raw)
            if dry_run:
                # A dry run wrote nothing, so the loader below cannot see what
                # was just fetched. Carry it in memory instead.
                fetched = ingested.candidates

    if ingest_only:
        # Deliberately before load_events: nothing downstream has run, so there
        # is no batch to finish and nothing to reconcile against.
        return result

    stored = events_repo.load_all()

    candidates = _stage(
        "load_candidates",
        lambda: candidates_repo.for_window(
            seen_since=now - timedelta(days=config.scraping.lookback_days),
            # The same floor ranking uses, so a candidate cannot be reloaded
            # into a batch that will discard the event it becomes. Not `now`:
            # ranking keeps an event that started at 00:30 on the run date, and
            # a 02:00 batch comparing against its own clock would drop it.
            starting_after=scope_floor(config, run_date),
        ),
        default=[],
    )
    candidates = _merge_candidates(candidates, fetched)
    result.stage_counts["candidates"] = len(candidates)

    norm_result = _stage(
        "normalization",
        lambda: normalization_service.run(candidates, get_now=get_now),
        default=None,
    )
    normalized = norm_result.events if norm_result else []
    _record_decisions(norm_result.decisions if norm_result else [])

    # The superseded ids are carried, not acted on. Reconcile knows which
    # duplicates lose hours before the run has anything to store in their place,
    # and deleting them here would leave a window across enrichment and
    # extraction with the duplicate gone and the winner unwritten. They ride
    # with the first save instead. `_carry_forward` already excludes them, so
    # nothing downstream sees them in the meantime.
    reconciled, stale_ids, merges = reconcile(normalized, stored)
    # Reconcile's delete was the pipeline's last destructive path. A cluster is
    # a labelled training scenario and a destroyed loser cannot be one, so a
    # displaced event is marked with what absorbed it and kept. No score: this
    # pass matches on shared candidate ids, not on similarity, and inventing a
    # number would misrepresent how the merge was decided.
    #
    # They stay out of the way on their own after this — `load_all` filters
    # superseded rows by default, so the next run never sees them at all.
    stored_by_id = {event.event_id: event for event in stored}
    displaced: list[Event] = []
    for loser_id, winner_id in merges.items():
        loser = stored_by_id.get(loser_id)
        if loser is None:
            continue
        loser.superseded_by = winner_id
        loser.superseded_at = now
        loser.merged_by = "reconcile"
        displaced.append(loser)

    in_scope = scope_filter(config, run_date, now)
    events = _carry_forward(reconciled, stored, stale_ids, in_scope)
    result.stage_counts["events"] = len(events)

    events = _stage(
        "enrichment", lambda: enrichment_service.enrich(events, run_date), default=events
    )
    # Extraction is the expensive stage — minutes an event on CPU — so it is
    # handed a way to persist as it goes. Without it, a run killed near the end
    # loses every model call it made. A dry run gets None: it persists nothing,
    # checkpoints included.
    extraction_stage.set_save_fn(None if dry_run else _save_one)
    # The same predicate ranking uses, for the reason the budget exists: model
    # time spent on an event ranking will discard buys nothing. It matters here
    # rather than in `_carry_forward` because a past event arrives on the
    # *fresh* side — `for_window` reloads every candidate discovered inside the
    # lookback whether or not its event is over — and the fresh side is
    # deliberately unscoped. Measured 2026-08-14: a whole 480-minute budget
    # went on events that had already happened.
    extraction_stage.set_scope_fn(in_scope)
    # The two stages that can run long enough to look dead. Cadence lives here
    # rather than in either stage: the stages report every item and decide
    # nothing, so how loud a run is stays one decision in one place — which is
    # what has to change the day extraction stops taking minutes an event.
    extraction_stage.set_progress_fn(_progress("extraction"))
    events = _stage("extraction", lambda: extraction_stage.process(events), default=events)
    # Read off the stage rather than counted here: an event with no hash may
    # have been deferred by the budget or refused by an unavailable provider,
    # and only the stage knows which. Not an error — the run ranks what it has —
    # but a count that stays high means the budget is set too low.
    result.stage_counts["extraction_deferred"] = extraction_stage.deferred
    # Read together: what the budget could not buy, and what it should never
    # have been asked to. Deferred alone cannot tell a backlog that is too big
    # from one that is full of events nothing can use.
    result.stage_counts["extraction_out_of_scope"] = extraction_stage.out_of_scope
    if not dry_run and events:
        # Wrapped like every other stage. Unwrapped, this line could end a batch
        # holding hours of extraction: it re-validates every event, so one bad
        # one discarded all the rest and the run never reached embedding,
        # scoring or ranking. Carrying on is a real recovery rather than a
        # shrug — embedding rebuilds whatever made the event unwritable, and
        # the save at the end of the run stores it. The stale ids simply go
        # undeleted until the next run reconciles them again.
        # Nothing is deleted: the displaced rows are written *alongside* the
        # live ones, in the same transaction, carrying their supersession.
        _stage("event persistence", lambda: events_repo.replace([], events + displaced))

    embedding_stage.set_progress_fn(_progress("embedding"))
    embedded = _stage("embedding", lambda: embedding_stage.process(events))
    if embedded is None:
        # Ranking on absent vectors produces meaningless order, and it would be
        # persisted as though it meant something. Stop instead.
        result.outcome = "failed"
        logger.error(
            "stopping before ranking: embedding produced nothing",
            component="batch",
            duration_ms=0,
        )
        return _finish()
    events = embedded
    _save(events)

    def _semantic_dedup() -> list[Event]:
        deduped = semantic_deduplicator.deduplicate(events, config.deduplication, now=now)
        _record_decisions(deduped.decisions)
        # The losers are saved, not dropped. They were already staying in the
        # database unmarked — unranked, unexplained, and findable only by
        # noticing an event that had never been scored. Saving them writes what
        # absorbed them; the repository keeps them out of ordinary reads.
        _save(deduped.superseded)
        return deduped.events

    events = _stage(
        "semantic_dedup",
        _semantic_dedup,
        default=events,
    )
    _save(events)

    events = _stage("similarity", lambda: similarity_stage.process(events), default=events)

    # Scoping, ranking and the order the two halves are written in all live in
    # the shared terminal step, so the read-time rescore cannot drift from this
    # one. What stays here is the batch's failure policy: `_stage` records the
    # error and carries on with a partial run.
    outcome = finish_run(
        events=events,
        run_date=run_date,
        now=now,
        config=config,
        ranking_engine=ranking_engine,
        score_repository=scores_repo,
        ranking_repository=rankings_repo,
        persist=not dry_run,
        run_stage=_stage,
    )
    result.stage_counts["ranked"] = outcome.rankable
    result.scores = outcome.scores
    result.rankings = outcome.rankings

    # Recorded before the refit reads them, so tonight's extractions are in
    # tonight's corpus. Append-only and keyed on the instant, so a retried run
    # cannot double-count itself.
    if not dry_run:
        _stage(
            "record_extractions",
            lambda: extraction_observation_repository.append(
                _extraction_observations(events, now)
            ),
        )

    # Last, and wrapped like every other stage. It reads rows this run has just
    # written and applies to the *next* one, so a failure here costs a night's
    # refit and nothing else — the ranking is already saved above.
    if not dry_run:
        result.refit = _stage(
            "refit",
            lambda: _refit(
                extraction_observation_repository, config, curve_state_repository, now
            ),
        )

    return _finish()


def _extraction_observations(
    events: list[Event], now: datetime
) -> list[ExtractionObservation]:
    """Tonight's extractions, as rows for the log.

    Stamped with the run's clock rather than the event's `updated_at`, because
    what the corpus needs is when the observation was made — the whole reason
    `created_at` could not be used is that it answers a different question.
    """
    return [
        ExtractionObservation(
            event_id=event.event_id,
            observed_at=now,
            chars=event.extraction_input_chars or 0,
            tags=len(event.tags),
            model=event.extraction_model,
            prompt_version=event.extraction_prompt_version,
            degradation=event.extraction_degradation,
            source=event.source,
        )
        for event in events
        if event.extraction_model is not None and event.extraction_input_chars
    ]


def _refit(
    observations_repo: ExtractionObservationRepository,
    config: AppConfig,
    curve_state: CurveStateRepository,
    now: datetime,
) -> str:
    """Re-derive the tag-confidence curve from what extraction actually did.

    Reads the incumbent from config, which composition has already replaced with
    whatever the last refit accepted — so the EWMA steps from where the run
    scored rather than from the file's defaults.

    Records the outcome whether or not it moved. "The gate said no" is part of
    the record: without it a night where nothing changed is indistinguishable
    from one where the refit never ran.
    """
    state = run_refit(
        observations_repo.load_all(),
        incumbent=(
            config.scoring.tag_confidence_cap,
            config.scoring.tag_confidence_saturation_chars,
        ),
        now=now,
    )
    if state is None:
        return "not run — no usable observations"

    curve_state.save(state)
    provenance = state.provenance
    rows = provenance.get("rows")
    if provenance.get("accepted"):
        return f"accepted — cap {state.cap:.3f}, saturation {state.saturation:.1f} ({rows} rows)"
    return f"refused — {provenance.get('reason')} ({rows} rows)"


def _merge_candidates(
    loaded: list[EventCandidate], fetched: list[EventCandidate]
) -> list[EventCandidate]:
    """Fold in candidates that were fetched but never written.

    The loaded ones win a collision, so the pipeline reads the same round-tripped
    objects a real run would. Sorted the way `load_candidates` sorts, so a dry
    run and a real run present the pipeline with the same order.
    """
    if not fetched:
        return loaded

    by_id = {c.id: c for c in fetched}
    by_id.update({c.id: c for c in loaded})
    return sorted(by_id.values(), key=lambda c: (c.discovered_at, c.id))


def _carry_forward(
    reconciled: list[Event],
    stored: list[Event],
    stale_ids: list[str],
    in_scope: Callable[[Event], bool],
) -> list[Event]:
    """Add the stored events that no fresh event claimed and are still rankable.

    Their candidates have aged out of the window, but the events have not: a
    calendar event found three weeks ago is still happening, and a night when
    every source is down should still re-rank what we already know about
    against tonight's forecast.

    Only the stored side is scoped, and only here. A stored event that has
    already happened would otherwise ride enrichment, dedup and similarity in
    full, just to be dropped by the same predicate immediately before ranking.
    Nothing is deleted — retention is #17's job, and a future feedback feature
    still wants the rows.

    Fresh events are never scoped here: extraction has not run yet, so their
    `start_time` is not knowable until later in the batch.
    """
    claimed = {e.event_id for e in reconciled} | set(stale_ids)
    return reconciled + [
        e for e in stored if e.event_id not in claimed and in_scope(e)
    ]


def _dedup_provenance(config: AppConfig) -> str:
    """The dedup constants in force, as JSON, for `run_history`.

    `scoring_config`'s argument applied to the other set of numbers that
    decides what a stored row means: a merge verdict is a function of
    thresholds that will be tuned, and a retuned threshold would otherwise
    reinterpret every decision already recorded.
    """
    return json.dumps(asdict(config.deduplication), sort_keys=True, default=str)


def _scoring_provenance(config: AppConfig) -> str:
    """The scoring constants in force, as JSON, for `run_history`.

    Serialised from the dataclass rather than re-read from the file, so it
    records what the run actually used — including any default that
    `config.yaml` never mentioned.
    """
    return json.dumps(asdict(config.scoring), sort_keys=True, default=str)


def _build_parser() -> argparse.ArgumentParser:
    """Flags are deliberately few; each earns its place.

    There is no `--force-extract`. Extraction's bypass is a property of the
    event, by design, with no special flag — deleting the row is the honest way
    to force one.
    """
    parser = argparse.ArgumentParser(
        prog="what-do-run-batch", description="Run the overnight batch."
    )
    parser.add_argument("--db", help=f"Path to the database (default: {DEFAULT_DB_PATH})")
    parser.add_argument("--config", help="Path to config.yaml")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Fetch and process everything, persist no events or recommendations",
    )
    parser.add_argument(
        "--skip-ingest",
        action="store_true",
        help="Re-run the pipeline over already-fetched candidates, touching no network",
    )
    parser.add_argument(
        "--ingest-only",
        action="store_true",
        help="Fetch from every source, report what each returned, and stop",
    )
    parser.add_argument(
        "--raw",
        nargs="?",
        const="-",
        metavar="PATH",
        help=(
            "Dump every fetched candidate as JSON Lines, with the reason any was "
            "discarded. Defaults to stdout; give a path to write a file"
        ),
    )
    parser.add_argument("--run-date", help="The date to run for, as YYYY-MM-DD")
    parser.add_argument(
        "--llm-transcript",
        nargs="?",
        const="-",
        metavar="PATH",
        help=(
            "Record every model call verbatim — full request and full response — "
            "as JSON Lines. Defaults to logs/llm-<timestamp>.jsonl; give a path "
            "to choose the file. Off unless asked for"
        ),
    )
    return parser


def _summarise(result: BatchResult) -> str:
    """Render what the run did, for a human reading cron output."""
    lines = [f"outcome: {result.outcome}"]
    for name, count in result.stage_counts.items():
        lines.append(f"  {name}: {count}")

    if result.per_source:
        lines.append("  sources:")
        width = max(len(name) for name in result.per_source)
        # Quietest first: the whole point of the table is to surface the source
        # that stopped producing, and it is never the one at the top of a
        # config file.
        for name, tally in sorted(
            result.per_source.items(), key=lambda item: (item[1].accepted, item[0])
        ):
            note = ""
            if tally.fetched and not tally.accepted:
                note = "   <- fetched but all discarded"
            elif not tally.fetched:
                note = "   <- nothing fetched"
            lines.append(
                f"    {name.ljust(width)}  {tally.accepted:>4} kept "
                f"of {tally.fetched:>4} fetched{note}"
            )

    lines.extend(_churn_lines(result.churn))
    lines.extend(_latch_lines(result.latch))
    if result.refit is not None:
        lines.append(f"  refit: {result.refit}")
    if result.failed_sources:
        lines.append(f"  failed sources: {', '.join(result.failed_sources)}")
    if result.skipped_sources:
        lines.append(f"  skipped sources: {', '.join(result.skipped_sources)}")
    for error in result.errors:
        lines.append(f"  error: {error}")
    return "\n".join(lines)


def _arm_identity_latches(
    churn: dict[str, ChurnTally],
    *,
    state: IdentityStateStore,
    config: AppConfig,
    db_path: Path,
    logger: StructuredLogger,
    get_now: Callable[[], datetime],
) -> LatchReport:
    """Bank this run's churn evidence, and act on it where it is decisive.

    A latch re-keys thousands of rows unattended, so a snapshot is taken first —
    the re-key verifies itself inside its transaction and rolls back on failure,
    but a rollback only covers what it anticipated.

    A failure here must not fail the run. The re-key wrote nothing if it raised,
    the latch survives in the database, and the next run tries again; losing a
    whole night's batch over it would be the worse outcome.
    """
    report = arm_latches(churn, state=state, sources=config.sources, at=get_now())

    for source in report.latched:
        logger.warning(
            f"identity latch: '{source}' re-mints its own ids, so it moves to "
            "content-derived ids permanently. Re-keying its stored candidates.",
            component="batch",
            duration_ms=0,
        )
        try:
            snapshot = snapshot_database(
                db_path, reason=f"latch-{source}", at=get_now()
            )
            logger.info(
                f"snapshot before re-key: {snapshot}", component="batch", duration_ms=0
            )
            outcome = rekey_to_content_ids(db_path, source=source)
            logger.info(
                f"re-keyed {source}: {outcome.candidates_before} rows became "
                f"{outcome.candidates_after}",
                component="batch",
                duration_ms=0,
            )
        except Exception as exc:  # noqa: BLE001 - reported, never fatal
            logger.error(
                f"re-key failed for '{source}': {exc}",
                component="batch",
                duration_ms=0,
            )

    for source in report.pinned_but_churning:
        logger.warning(
            f"'{source}' is pinned to its publisher's ids and is churning "
            "anyway. The pin suppresses the latch, not the measurement.",
            component="batch",
            duration_ms=0,
        )

    return report


def _latch_lines(report: LatchReport) -> list[str]:
    """Announce a latch loudly. It changes a source's identity permanently."""
    lines: list[str] = []
    for source in report.latched:
        lines.append(
            f"  identity latch: {source} moved to content-derived ids "
            "(its own ids re-mint)"
        )
    for source in report.pinned_but_churning:
        lines.append(
            f"  identity: {source} is pinned to publisher ids and churning anyway"
        )
    return lines


def _churn_lines(churn: dict[str, ChurnTally]) -> list[str]:
    """Report only what was actually measured, and never a bare silence.

    A source with nothing seen before has an *undefined* rate, and rendering it
    as 0.00 would claim its ids are stable on no evidence — the reading that
    would have blessed northshorenightout on its first night.

    Nothing at all is worse still: a run where the measurement did not happen
    would look exactly like a run where every source behaved, which is how the
    refit went a night without anybody noticing.
    """
    measured = {name: t for name, t in churn.items() if t.rate is not None}
    if not measured:
        return []

    unstable = sorted(
        ((name, t) for name, t in measured.items() if t.rate),
        key=lambda item: -(item[1].rate or 0),
    )
    if not unstable:
        return [f"  id churn: none — {len(measured)} source(s) kept their ids"]

    lines = ["  id churn:"]
    width = max(len(name) for name, _ in unstable)
    lines.extend(
        f"    {name.ljust(width)}  {tally.churned:>4} of {tally.seen_before:>4} "
        f"known listings re-minted ({(tally.rate or 0):.0%})"
        for name, tally in unstable
    )
    return lines


def _write_raw(records: list[Any], destination: str, stdout: TextIO) -> None:
    """Dump every fetched candidate as JSON Lines.

    One object per line rather than one array, so a dump of thousands stays
    greppable and a truncated file is still readable to its last line.
    """
    stream = stdout if destination == "-" else open(destination, "w", encoding="utf-8")
    try:
        for record in records:
            payload = {
                "source": record.source,
                "verdict": record.verdict,
                "reason": record.reason,
                "candidate": {
                    key: (value.isoformat() if isinstance(value, datetime) else value)
                    for key, value in asdict(record.candidate).items()
                },
            }
            print(json.dumps(payload, ensure_ascii=False), file=stream)
    finally:
        if stream is not stdout:
            stream.close()


def run(
    argv: list[str] | None = None,
    *,
    get_now: Callable[[], datetime] = _default_now,
    stdout: TextIO = sys.stdout,
    stderr: TextIO = sys.stderr,
    load_config_fn: Callable[..., AppConfig] = load_config,
    build_dependencies_fn: Callable[..., BatchDependencies] = build_dependencies,
    run_batch_fn: Callable[..., BatchResult] = run_batch,
    init_db_fn: Callable[[Path], None] = init_db,
    check_schema_fn: Callable[[Path], list[Finding]] = check_database,
    check_config_fn: Callable[[AppConfig, Path], list[ConfigFinding]] = check_config_file,
    transcript_factory: Callable[..., Any] = LLMTranscript,
) -> int:
    """Entry point for `what-do-run-batch`.

    Args:
        argv: Command-line arguments, defaulting to `sys.argv[1:]`.
        get_now: Injectable clock; nothing in this project reads one directly.
        stdout: Where the run summary goes.
        stderr: Where argument errors go.
        load_config_fn: Injected for testing, as the CLI injects its loaders.
        build_dependencies_fn: Injected for testing.
        run_batch_fn: Injected for testing.
        init_db_fn: Injected for testing.
        check_schema_fn: Injected for testing. Defaults to the real check,
            which is asserted separately against a database on disk — a seam
            only tests ever reach is a seam production never exercises.
        check_config_fn: Injected for testing, on the same terms. Reports
            features left switched off; never fails the run.
        transcript_factory: Builds the LLM transcript. Injected so a test can
            assert the file chosen without one being opened.

    Returns:
        Process exit code. A `partial` run still exits zero — a stage failed but
        recommendations were produced. Only `failed` is non-zero, which is the
        case where the batch stopped before ranking and cron should notice.
    """
    args = _build_parser().parse_args(argv)

    run_date = get_now().date()
    if args.run_date:
        try:
            run_date = date.fromisoformat(args.run_date)
        except ValueError:
            print(f"Invalid --run-date {args.run_date!r}: expected YYYY-MM-DD", file=stderr)
            return 2

    db_path = Path(args.db) if args.db else DEFAULT_DB_PATH
    config = load_config_fn(Path(args.config) if args.config else None)
    logger = get_logger("batch")

    # Schema, not data: the caches the pipeline reads through need their tables
    # to exist even on a dry run, which promises to write no events, no
    # recommendations, and no deletes.
    init_db_fn(db_path)

    # After healing, before anything is built. `_SCHEMA` is all CREATE TABLE IF
    # NOT EXISTS, so `init_db` above adds a missing *table* and silently skips a
    # missing *column*: every fresh database — every test — has it and the live
    # file does not. This is the only check that can tell.
    #
    # It aborts rather than reports, and applies to `--dry-run` and
    # `--ingest-only` alike. A drift that reaches a stage kills the run minutes
    # later anyway, having burnt model time first; and a dry run that passes
    # against a schema the real run would die on gives exactly the false
    # confidence it exists to prevent.
    schema_findings = check_schema_fn(db_path)
    if schema_findings:
        print(format_findings(db_path, schema_findings), file=stdout)
        return 1

    transcript = None
    if args.llm_transcript is not None:
        transcript_path = (
            DEFAULT_LOG_DIR / f"llm-{get_now().strftime('%Y%m%d-%H%M%S')}.jsonl"
            if args.llm_transcript == "-"
            else Path(args.llm_transcript)
        )
        transcript = transcript_factory(transcript_path, get_now=get_now)

    dependencies = build_dependencies_fn(
        config=config,
        db_path=db_path,
        seeds_path=DEFAULT_SEEDS_PATH,
        likes_path=DEFAULT_LIKES_PATH,
        dislikes_path=DEFAULT_DISLIKES_PATH,
        blocklist_path=DEFAULT_BLOCKLIST_PATH,
        logger=logger,
        get_now=get_now,
        llm_transcript=transcript,
    )

    try:
        result = run_batch_fn(
            config=config,
            db_path=db_path,
            ingestion_service=dependencies.ingestion_service,
            normalization_service=dependencies.normalization_service,
            enrichment_service=dependencies.enrichment_service,
            extraction_stage=dependencies.extraction_stage,
            embedding_stage=dependencies.embedding_stage,
            semantic_deduplicator=dependencies.semantic_deduplicator,
            similarity_stage=dependencies.similarity_stage,
            ranking_engine=dependencies.ranking_engine,
            candidate_repository=dependencies.candidate_repository,
            event_repository=dependencies.event_repository,
            run_repository=dependencies.run_repository,
            score_repository=dependencies.score_repository,
            ranking_repository=dependencies.ranking_repository,
            dedup_decision_repository=dependencies.dedup_decision_repository,
            curve_state_repository=dependencies.curve_state_repository,
            extraction_observation_repository=dependencies.extraction_observation_repository,
            preference_revision_repository=dependencies.preference_revision_repository,
            identity_state_repository=dependencies.identity_state_repository,
            preference_revision=dependencies.preference_revision,
            heartbeat_path=HEARTBEAT_PATH,
            logger=logger,
            run_date=run_date,
            get_now=get_now,
            skipped_sources=dependencies.skipped_sources,
            skip_ingest=args.skip_ingest,
            dry_run=args.dry_run,
            ingest_only=args.ingest_only,
            raw_dump_fn=(
                None
                if args.raw is None
                else lambda records: _write_raw(records, args.raw, stdout)
            ),
        )
    finally:
        # A batch that died is exactly when the transcript matters, so it is
        # flushed and closed on the way out either way.
        if transcript is not None:
            transcript.close()

    print(_summarise(result), file=stdout)
    # After the summary, and never fatal. A fresh clone configures almost
    # nothing and must still produce a batch — the schema check is the one that
    # refuses. This only answers "did tonight run with a feature switched off",
    # which went unasked for twelve days while every weather adjustment was 0.0.
    config_path = Path(args.config) if args.config else DEFAULT_CONFIG_PATH
    for line in _config_check_lines(check_config_fn(config, config_path)):
        print(line, file=stdout)
    return 1 if result.outcome == "failed" else 0


def _config_check_lines(findings: list[ConfigFinding]) -> list[str]:
    """The switched-off features, or nothing at all when there are none."""
    if not findings:
        return []
    width = max(len(finding.path) for finding in findings)
    return [
        "  config:",
        *[
            f"    {finding.level:<7} {finding.path.ljust(width)}  {finding.detail}"
            for finding in findings
        ],
    ]
