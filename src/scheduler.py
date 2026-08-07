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
import sys
import zoneinfo
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Callable, TextIO

from src.composition import BatchDependencies, build_dependencies
from src.config import AppConfig, load_config
from src.enrichment.service import EnrichmentService
from src.ingestion.ingestion_service import IngestionService
from src.models.event import Event
from src.models.event_candidate import EventCandidate
from src.models.recommendation import Recommendation
from src.normalization.reconcile import reconcile
from src.normalization.semantic_dedup import SemanticDeduplicationEngine
from src.normalization.service import NormalizationService
from src.processing.extraction_stage import ExtractionStage
from src.scoring.embedding_stage import EmbeddingStage
from src.scoring.ranking import RankingEngine
from src.scoring.similarity_stage import SimilarityStage
from src.storage.candidates import load_candidates
from src.storage.db import DEFAULT_DB_PATH, init_db
from src.storage.events import delete_events, load_events, save_events
from src.storage.recommendations import save_recommendations
from src.storage.runs import finish_run, start_run
from src.utils.logging import StructuredLogger, get_logger

DEFAULT_SEEDS_PATH = Path("data/seeds.yaml")
DEFAULT_LIKES_PATH = Path("data/likes.txt")
DEFAULT_DISLIKES_PATH = Path("data/dislikes.txt")
DEFAULT_BLOCKLIST_PATH = Path("data/blocklist.json")


@dataclass
class BatchResult:
    """What one run of the batch did."""

    outcome: str
    stage_counts: dict[str, int] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    skipped_sources: list[str] = field(default_factory=list)
    recommendations: list[Recommendation] = field(default_factory=list)


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
    get_now: Callable[[], datetime] = datetime.now,
    skipped_sources: list[str] | None = None,
    skip_ingest: bool = False,
    dry_run: bool = False,
    load_candidates_fn: Callable[..., list[EventCandidate]] = load_candidates,
    load_events_fn: Callable[..., list[Event]] = load_events,
    save_events_fn: Callable[..., None] = save_events,
    delete_events_fn: Callable[..., None] = delete_events,
    save_recommendations_fn: Callable[..., None] = save_recommendations,
    start_run_fn: Callable[..., str] = start_run,
    finish_run_fn: Callable[..., None] = finish_run,
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
        load_candidates_fn: Injected for testing, as the CLI injects its loaders.
        load_events_fn: Injected for testing.
        save_events_fn: Injected for testing.
        delete_events_fn: Injected for testing.
        save_recommendations_fn: Injected for testing.
        start_run_fn: Injected for testing.
        finish_run_fn: Injected for testing.

    Returns:
        BatchResult describing the outcome, per-stage counts, and any errors.
    """
    result = BatchResult(outcome="success", skipped_sources=list(skipped_sources or []))
    now = get_now()

    # A dry run did not do a batch, so recording one would pollute the only
    # durable record of what the nightly runs actually did.
    run_id = None if dry_run else start_run_fn(db_path, now)

    def _finish() -> BatchResult:
        """Complete the history row and hand back the result.

        Deliberately not a `finally`: a run that dies outside the stage
        wrappers should leave its row with a `started_at` and no
        `completed_at`, because that is what a crash looks like. Marking it
        finished on the way out would erase the one signal the start-row
        exists to give.
        """
        if run_id is not None:
            finish_run_fn(
                db_path,
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
            save_events_fn(events, db_path)

    fetched: list[EventCandidate] = []
    if skip_ingest:
        logger.info("skipping ingestion", component="batch", duration_ms=0)
    else:
        ingested = _stage(
            "ingestion",
            lambda: ingestion_service.run(get_now=get_now, persist=not dry_run),
        )
        if ingested is not None:
            result.stage_counts["ingested"] = ingested.accepted
            if dry_run:
                # A dry run wrote nothing, so the loader below cannot see what
                # was just fetched. Carry it in memory instead.
                fetched = ingested.candidates

    stored = load_events_fn(db_path)

    candidates = _stage(
        "load_candidates",
        lambda: load_candidates_fn(
            db_path,
            discovered_since=now - timedelta(days=config.scraping.lookback_days),
            starting_after=now,
        ),
        default=[],
    )
    candidates = _merge_candidates(candidates, fetched)
    result.stage_counts["candidates"] = len(candidates)

    normalized = _stage(
        "normalization",
        lambda: normalization_service.run(candidates, get_now=get_now).events,
        default=[],
    )

    reconciled, stale_ids = reconcile(normalized, stored)
    if stale_ids and not dry_run:
        delete_events_fn(stale_ids, db_path)

    in_scope = _scope_filter(config, run_date, now)
    events = _carry_forward(reconciled, stored, stale_ids, in_scope)
    result.stage_counts["events"] = len(events)

    events = _stage(
        "enrichment", lambda: enrichment_service.enrich(events, run_date), default=events
    )
    events = _stage("extraction", lambda: extraction_stage.process(events), default=events)
    _save(events)

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

    events = _stage(
        "semantic_dedup",
        lambda: semantic_deduplicator.deduplicate(events, config.deduplication),
        default=events,
    )
    _save(events)

    events = _stage("similarity", lambda: similarity_stage.process(events), default=events)

    rankable = [e for e in events if in_scope(e)]
    result.stage_counts["ranked"] = len(rankable)

    recommendations = _stage(
        "ranking", lambda: ranking_engine.rank(rankable, run_date), default=[]
    )
    result.recommendations = recommendations

    if not dry_run and recommendations:
        _stage("save_recommendations", lambda: save_recommendations_fn(recommendations, db_path))

    return _finish()


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


def _scope_filter(
    config: AppConfig, run_date: date, now: datetime
) -> Callable[[Event], bool]:
    """Build the predicate for whether an event is worth ranking this run.

    Ranking everything ever stored grows without bound; ranking only tonight
    discards the lookahead the calendar feeds exist for. Undated events are kept
    on discovery age instead, because the CLI has a labelled section for them
    and dropping one would lose a real event to a failed extraction.
    """
    horizon = run_date + timedelta(days=config.scraping.horizon_days)
    lookback_cutoff = now - timedelta(days=config.scraping.lookback_days)
    tz = zoneinfo.ZoneInfo(config.location.timezone)

    def in_scope(event: Event) -> bool:
        if event.start_time is None:
            return event.created_at >= lookback_cutoff
        # Local date, not UTC: an event at 11pm local is tomorrow in UTC, and
        # filtering on that would misfile exactly the evening events we rank.
        start = event.start_time.astimezone(tz).date()
        return run_date <= start <= horizon

    return in_scope


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
    parser.add_argument("--run-date", help="The date to run for, as YYYY-MM-DD")
    return parser


def _summarise(result: BatchResult) -> str:
    """Render what the run did, for a human reading cron output."""
    lines = [f"outcome: {result.outcome}"]
    for name, count in result.stage_counts.items():
        lines.append(f"  {name}: {count}")
    if result.skipped_sources:
        lines.append(f"  skipped sources: {', '.join(result.skipped_sources)}")
    for error in result.errors:
        lines.append(f"  error: {error}")
    return "\n".join(lines)


def run(
    argv: list[str] | None = None,
    *,
    get_now: Callable[[], datetime] = datetime.now,
    stdout: TextIO = sys.stdout,
    stderr: TextIO = sys.stderr,
    load_config_fn: Callable[..., AppConfig] = load_config,
    build_dependencies_fn: Callable[..., BatchDependencies] = build_dependencies,
    run_batch_fn: Callable[..., BatchResult] = run_batch,
    init_db_fn: Callable[[Path], None] = init_db,
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

    dependencies = build_dependencies_fn(
        config=config,
        db_path=db_path,
        seeds_path=DEFAULT_SEEDS_PATH,
        likes_path=DEFAULT_LIKES_PATH,
        dislikes_path=DEFAULT_DISLIKES_PATH,
        blocklist_path=DEFAULT_BLOCKLIST_PATH,
        logger=logger,
        get_now=get_now,
    )

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
        logger=logger,
        run_date=run_date,
        get_now=get_now,
        skipped_sources=dependencies.skipped_sources,
        skip_ingest=args.skip_ingest,
        dry_run=args.dry_run,
    )

    print(_summarise(result), file=stdout)
    return 1 if result.outcome == "failed" else 0
