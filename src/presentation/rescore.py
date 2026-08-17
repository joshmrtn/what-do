"""Recompute a stale ranking before showing it.

The read path's whole reason for existing: extraction is the only stage too slow
to run at query time, so everything else can be re-run against fresher inputs
and the listing on screen is a real computation rather than a patched number.

Nothing here decides *whether* to run — `rescore_policy` does that, and most
invocations answer no. What this owns is the recomputation and, more
importantly, the rule that it may never cost the listing: any failure at all
returns None and the caller shows what was stored.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Callable

from src.composition.pipeline import run_view_tail
from src.composition.storage import ViewStorage
from src.composition.view import build_rescore_pipeline
from src.config import AppConfig
from src.enrichment.weather import WeatherProvider
from src.models.preference_revision import PreferenceRevision
from src.models.rescore import Rescore
from src.presentation.filters import RankedEvent
from src.presentation.freshness import latest_forecast
from src.presentation.rescore_policy import (
    RESCORE,
    batch_is_running,
    should_rescore,
)
from src.storage.queries import load_ranked_events
from src.utils.logging import StructuredLogger


def rescore_if_stale(
    *,
    pairs: list[RankedEvent],
    tonight: date,
    now: datetime,
    ttl: timedelta,
    config: AppConfig,
    db_path: Path,
    storage: ViewStorage,
    logger: StructuredLogger,
    get_now: Callable[[], datetime],
    weather_provider: WeatherProvider,
    likes_path: Path,
    dislikes_path: Path,
    blocklist_path: Path,
) -> list[RankedEvent] | None:
    """Recompute the listing against a fresh forecast, or leave it alone.

    Args:
        pairs: The listing as loaded, in the batch's order.
        tonight: The night being shown, in the view's own zone.
        now: The current instant.
        ttl: How old the forecast may be before it is worth refreshing.
        config: Loaded application config.
        db_path: Database, for the weather cache and preference vectors.
        storage: The view's repositories — read from and, here only, written to.
        logger: Structured logger.
        get_now: Injectable clock, handed to enrichment.
        weather_provider: Where the fresh forecast comes from.
        likes_path: Preference file.
        dislikes_path: The same, for dislikes.
        blocklist_path: Venue names never to surface.

    Returns:
        The recomputed listing, or None when nothing was done — which covers
        both "no reason to" and "it was tried and failed". The caller renders
        what it already has either way.
    """
    if not pairs:
        return None

    open_run = storage.runs.open_run()
    decision = should_rescore(
        run_date=pairs[0].ranking.run_date,
        tonight=tonight,
        forecast_issued_at=latest_forecast(pair.event for pair in pairs),
        now=now,
        ttl=ttl,
        batch_running=batch_is_running(
            open_run.started_at if open_run is not None else None, now
        ),
    )
    if decision != RESCORE:
        return None

    run_date = pairs[0].ranking.run_date
    try:
        # Building the pipeline is itself a place this can decline: loading
        # preferences through a refusing embedding provider raises when a line
        # has no cached vector, which is exactly the edited-`likes.txt` case.
        pipeline = build_rescore_pipeline(
            config=config,
            db_path=db_path,
            logger=logger,
            get_now=get_now,
            storage=storage,
            weather_provider=weather_provider,
            likes_path=likes_path,
            dislikes_path=dislikes_path,
            blocklist_path=blocklist_path,
        )
        outcome = run_view_tail(
            events=[pair.event for pair in pairs],
            run_date=run_date,
            now=now,
            config=config,
            enrichment_service=pipeline.enrichment_service,
            embedding_stage=pipeline.embedding_stage,
            similarity_stage=pipeline.similarity_stage,
            ranking_engine=pipeline.ranking_engine,
            event_repository=storage.events,
            score_repository=storage.scores,
            ranking_repository=storage.rankings,
            forecast_fresh_since=now - ttl,
        )
        if not outcome.rankings:
            # Writing nothing is the correct outcome of a failed rank, and
            # `finish_run` already declined to persist it. Saying so here stops
            # a caller reading the empty result as a real, empty listing.
            return None

        refreshed = load_ranked_events(
            storage.events, storage.scores, storage.rankings, run_date=run_date
        )
        _record(storage, run_date, refreshed, now)
        return refreshed
    except Exception as exc:  # noqa: BLE001 — a rescore must never cost the listing
        logger.warning(
            f"rescore abandoned, showing the stored ranking: {exc}",
            component="rescore",
            duration_ms=0,
        )
        return None


def _record(
    storage: ViewStorage,
    run_date: date,
    refreshed: list[RankedEvent],
    now: datetime,
) -> None:
    """Write down that this run's ordering is no longer the batch's.

    Appended, never updated. Without it, a run date holds numbers produced by a
    forecast its own `run_history` row does not describe, and nothing anywhere
    says so.
    """
    revision: PreferenceRevision | None = storage.preference_revisions.latest()
    # Re-recording is how the id is recovered: the write is keyed on content, so
    # it resolves to the existing row rather than creating one. And the content
    # cannot be new here — a preference line without a cached vector would have
    # raised out of `build_rescore_pipeline` long before this point.
    revision_id = (
        None if revision is None else storage.preference_revisions.record(revision)
    )
    storage.rescores.record(
        Rescore(
            run_date=run_date,
            rescored_at=now,
            events_rescored=len(refreshed),
            forecast_issued_at=latest_forecast(pair.event for pair in refreshed),
            preference_revision_id=revision_id,
        )
    )
