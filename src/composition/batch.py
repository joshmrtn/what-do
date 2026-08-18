"""Composition root: the one place real providers are constructed.

Every stage in this project takes its dependencies injected, which is what lets
the tests substitute fakes and never touch the network. The consequence is that
nothing, anywhere, actually built a real adapter — until here.

Credential policy: a source whose key is absent is skipped with a warning that
names the exact variable it looked for, and the skip is reported rather than
inferred downstream. It never marks the run partial. A user who legitimately
has no Apify key would otherwise be `partial` every night forever, and an
outcome that never varies carries no information; `partial` is for stage
failures. A *wrong* key is not silent under this policy either — the adapter
builds normally and fails at its first request with a 401, which is an ordinary
stage failure. That leaves a mistyped variable *name* as the one genuinely
silent case, which is why the warning names the variable and not just the
source.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Mapping

import requests

from src.composition.network import (
    build_air_quality_provider,
    build_movie_provider,
    build_http_fetcher,
    build_request_policy,
    build_weather_provider,
)
from src.config import AppConfig
from src.enrichment.astronomical import AstronomicalCalculator
from src.enrichment.service import EnrichmentService
from src.ingestion.aggregators.do617_source import Do617VenueSource
from src.ingestion.aggregators.jsonld_source import JsonLdEventSource
from src.ingestion.calendars.assabet_source import AssabetRssSource
from src.ingestion.calendars.html_source import HtmlListingSource
from src.ingestion.calendars.ics_source import IcsCalendarSource
from src.ingestion.identity import content_id_rule
from src.ingestion.calendars.moon_source import MoonRssSource
from src.ingestion.calendars.tribe_source import TribeCalendarSource
from src.ingestion.cinemas.cabot_source import CabotListingSource
from src.ingestion.cinemas.veezi_source import VeeziSessionsSource
from src.ingestion.ingestion_service import IngestionService
from src.ingestion.movies.amc import AmcAdapter
from src.ingestion.seeds import load_seeds
from src.ingestion.social.apify import ApifyAdapter
from src.ingestion.social.dumpor import DumporAdapter
from src.ingestion.social.picuki import PicukiAdapter
from src.normalization.semantic_dedup import SemanticDeduplicationEngine
from src.normalization.service import NormalizationService
from src.processing.extraction import OllamaExtractionProvider
from src.processing.extraction_stage import ExtractionStage
from src.processing.image_fetcher import HttpImageFetcher
from src.scoring.embedding_stage import EmbeddingStage
from src.scoring.embeddings import OllamaEmbeddingProvider
from src.models.preference_revision import PreferenceRevision
from src.scoring.preference_revision import build_revision
from src.scoring.preferences import PreferenceRepository
from src.scoring.ranking import RankingEngine
from src.scoring.similarity_stage import SimilarityStage
import dataclasses

from src.composition.pipeline import load_blocklist
from src.composition.storage import build_batch_storage
from src.storage.protocols import (
    CurveStateRepository,
    ExtractionObservationRepository,
    DedupDecisionRepository,
    CandidateRepository,
    EntityRepository,
    EventRepository,
    PreferenceRevisionRepository,
    RankingRepository,
    RunRepository,
    ScoreRepository,
)
from src.storage.events import load_tag_embeddings
from src.utils.logging import StructuredLogger
from src.utils.llm_transcript import TranscriptSink
from src.utils.ollama_client import OllamaClient


@dataclass
class BatchDependencies:
    """Everything `run_batch` needs, built from config and the environment."""

    ingestion_service: IngestionService
    normalization_service: NormalizationService
    enrichment_service: EnrichmentService
    extraction_stage: ExtractionStage
    embedding_stage: EmbeddingStage
    semantic_deduplicator: SemanticDeduplicationEngine
    similarity_stage: SimilarityStage
    ranking_engine: RankingEngine
    skipped_sources: list[str]
    #: Persistence. Named here and nowhere else: `run_batch` used to fall back
    #: to constructing its own SQLite repository whenever one was not injected,
    #: so a caller who forgot silently got a database anyway — the one thing the
    #: repository split exists to prevent.
    candidate_repository: CandidateRepository
    event_repository: EventRepository
    run_repository: RunRepository
    score_repository: ScoreRepository
    ranking_repository: RankingRepository
    dedup_decision_repository: DedupDecisionRepository
    curve_state_repository: CurveStateRepository
    extraction_observation_repository: ExtractionObservationRepository
    preference_revision_repository: PreferenceRevisionRepository
    #: What the preference files said when this run loaded them. Built here
    #: because this is where they are read; `run_batch` records it rather
    #: than rebuilding it, so the run stamps the revision it actually scored
    #: against.
    preference_revision: PreferenceRevision


def build_dependencies(
    *,
    config: AppConfig,
    db_path: Path,
    seeds_path: Path,
    likes_path: Path,
    dislikes_path: Path,
    blocklist_path: Path,
    logger: StructuredLogger,
    get_now: Callable[[], datetime] = datetime.now,
    env: Mapping[str, str] | None = None,
    llm_transcript: TranscriptSink | None = None,
) -> BatchDependencies:
    """Construct every real provider and stage for one batch run.

    Args:
        config: Loaded application config.
        db_path: Path to the SQLite database.
        seeds_path: Path to seeds.yaml.
        likes_path: Path to likes.txt.
        dislikes_path: Path to dislikes.txt.
        blocklist_path: Path to blocklist.json.
        logger: Structured logger.
        get_now: Injectable clock, passed to every adapter that needs one.
        env: Environment mapping. Injected so tests never read the developer's
            real `.env` — see #6.
        llm_transcript: Where every model call is recorded verbatim. None, the
            default, records nothing.

    Returns:
        BatchDependencies, including the sources skipped for a missing key.
    """
    environ: Mapping[str, str] = os.environ if env is None else env
    skipped: list[str] = []

    def _credential(variable: str, source: str) -> str | None:
        """Return a usable credential, or record the skip and return None."""
        value = (environ.get(variable) or "").strip()
        if value:
            return value
        # Naming the variable is the point: an absent key is a legitimate
        # deployment state, so the only genuinely silent failure left is a
        # mistyped variable name.
        logger.warning(
            f"skipping source '{source}': {variable} is not set",
            component="composition",
            duration_ms=0,
        )
        skipped.append(source)
        return None

    storage = build_batch_storage(db_path, config.models.embeddings)
    # The curve the *last* refit accepted, applied before anything is scored.
    # Reading it here rather than mid-run is the point: a night is scored with
    # constants that did not move underneath it.
    config = _with_accepted_curve(config, storage.curve_state, logger)
    handles = _handles(storage.entities, seeds_path)
    blocklist = load_blocklist(blocklist_path)

    # One fetcher, shared by every adapter. It owns the session, the throttle,
    # the retry schedule and the cache, so an adapter no longer holds any of
    # them — and there is one place that can answer what we asked of whom.
    #
    # The throttle is shared for the same reason the cache is: two adapters
    # pointed at one host are one conversation from the server's side.
    request_policy = build_request_policy(config, get_now=get_now, logger=logger)
    fetcher = build_http_fetcher(
        config,
        http_cache=storage.http_cache,
        get_now=get_now,
        policy=request_policy,
        logger=logger,
    )

    # Whether each source's publisher may be trusted to identify its own
    # listings. One rule, built once and handed to every adapter that keys a
    # candidate on something the publisher supplied — so there is one place the
    # answer can come from when the churn latch starts supplying it too.
    uses_content_id = content_id_rule(config.sources)

    # Alternative routes to the same Instagram data, tried in order. Apify is
    # first because it is the only one under contract; the scrapers are the
    # fallback when it is unavailable or unaffordable.
    failover_sources: list[Any] = []
    apify_key = _credential("APIFY_API_KEY", "apify")
    if apify_key:
        failover_sources.append(
            ApifyAdapter(apify_key, handles, fetcher, get_now=get_now)
        )
    failover_sources.append(PicukiAdapter(handles, fetcher, get_now=get_now))
    failover_sources.append(DumporAdapter(handles, fetcher, get_now=get_now))

    independent_sources: list[Any] = []
    amc_key = _credential("AMC_API_KEY", "amc")
    if amc_key:
        independent_sources.append(
            AmcAdapter(
                amc_key,
                config.location.postal_code,
                session=requests.Session(),
                policy=request_policy,
                get_now=get_now,
                uses_content_id=uses_content_id,
            )
        )
    for feed in config.sources.ics_calendars:
        independent_sources.append(
            IcsCalendarSource(
                feed,
                fetcher,
                get_now=get_now,
                logger=logger,
                timezone_name=config.location.timezone,
                horizon_days=config.scraping.horizon_days,
                day_starts_at=config.day_starts_at,
                uses_content_id=uses_content_id,
            )
        )
    for feed in config.sources.veezi_cinemas:
        independent_sources.append(
            VeeziSessionsSource(
                feed,
                fetcher,
                get_now=get_now,
                logger=logger,
                timezone_name=config.location.timezone,
                uses_content_id=uses_content_id,
            )
        )
    for feed in config.sources.tribe_calendars:
        independent_sources.append(
            TribeCalendarSource(
                feed,
                fetcher,
                get_now=get_now,
                logger=logger,
                timezone_name=config.location.timezone,
                horizon_days=config.scraping.horizon_days,
                day_starts_at=config.day_starts_at,
                uses_content_id=uses_content_id,
            )
        )
    for feed in config.sources.cabot_listings:
        independent_sources.append(
            CabotListingSource(
                feed,
                fetcher,
                get_now=get_now,
                logger=logger,
                timezone_name=config.location.timezone,
                horizon_days=config.scraping.horizon_days,
                day_starts_at=config.day_starts_at,
                uses_content_id=uses_content_id,
            )
        )
    for feed in config.sources.do617_venues:
        independent_sources.append(
            Do617VenueSource(
                feed,
                fetcher,
                get_now=get_now,
                logger=logger,
                timezone_name=config.location.timezone,
                horizon_days=config.scraping.horizon_days,
                day_starts_at=config.day_starts_at,
            )
        )
    for feed in config.sources.jsonld_pages:
        independent_sources.append(
            JsonLdEventSource(
                feed,
                fetcher,
                get_now=get_now,
                logger=logger,
                timezone_name=config.location.timezone,
                horizon_days=config.scraping.horizon_days,
                day_starts_at=config.day_starts_at,
            )
        )
    for feed in config.sources.assabet_feeds:
        independent_sources.append(
            AssabetRssSource(
                feed,
                fetcher,
                get_now=get_now,
                logger=logger,
                timezone_name=config.location.timezone,
                horizon_days=config.scraping.horizon_days,
                day_starts_at=config.day_starts_at,
            )
        )
    for feed in config.sources.moon_feeds:
        independent_sources.append(
            MoonRssSource(
                feed,
                fetcher,
                get_now=get_now,
                logger=logger,
                timezone_name=config.location.timezone,
                horizon_days=config.scraping.horizon_days,
                day_starts_at=config.day_starts_at,
            )
        )
    for feed in config.sources.html_calendars:
        independent_sources.append(
            HtmlListingSource(
                feed,
                fetcher,
                config.location.timezone,
                get_now=get_now,
                logger=logger,
            )
        )

    # Two clients over one host: they differ only in the name their calls carry
    # into the transcript, which is what makes a slow extraction distinguishable
    # from the hundreds of embedding calls around it.
    extraction_client = OllamaClient(
        config.ollama_host,
        timeout=config.models.request_timeout_seconds,
        transcript=llm_transcript,
        component="extraction",
        options={
            "temperature": config.models.temperature,
            "top_p": config.models.top_p,
            "num_ctx": config.models.num_ctx,
        },
        think=config.models.think,
        response_format=config.models.response_format,
        keep_alive=config.models.keep_alive,
    )
    # No chat parameters: /api/embed takes none of them, and the embedding
    # model neither samples nor reasons.
    embedding_client = OllamaClient(
        config.ollama_host,
        timeout=config.models.request_timeout_seconds,
        transcript=llm_transcript,
        component="embedding",
        keep_alive=config.models.keep_alive,
    )
    embedding_provider = OllamaEmbeddingProvider(
        embedding_client, model=config.models.embeddings
    )

    tmdb_key = _credential("TMDB_API_KEY", "tmdb")

    preferences = PreferenceRepository(embedding_provider, db_path, logger).load(
        likes_path, dislikes_path
    )
    preference_revision = build_revision(
        preferences,
        likes_name=likes_path.name,
        dislikes_name=dislikes_path.name,
        captured_at=get_now(),
    )

    return BatchDependencies(
        ingestion_service=IngestionService(
            config=config,
            db_path=db_path,
            seeds_path=seeds_path,
            failover_sources=failover_sources,
            independent_sources=independent_sources,
            logger=logger,
            entities=storage.entities,
            blocklist=blocklist,
        ),
        normalization_service=NormalizationService(config, logger),
        enrichment_service=EnrichmentService(
            weather_provider=build_weather_provider(
                config,
                weather_cache=storage.weather_cache,
                get_now=get_now,
                policy=request_policy,
                logger=logger,
            ),
            movie_provider=(
                build_movie_provider(
                    config,
                    tmdb_key,
                    movie_cache=storage.movie_cache,
                    get_now=get_now,
                    policy=request_policy,
                    logger=logger,
                )
                if tmdb_key
                else None
            ),
            astronomical_calculator=AstronomicalCalculator(),
            synthetic_rules=config.synthetic_activities,
            config=config,
            db_path=db_path,
            air_quality_provider=build_air_quality_provider(
                config,
                air_quality_cache=storage.air_quality_cache,
                get_now=get_now,
                policy=request_policy,
                logger=logger,
            ),
            get_now=get_now,
            logger=logger,
        ),
        extraction_stage=ExtractionStage(
            provider=OllamaExtractionProvider(
                extraction_client,
                model=config.models.llm_extraction,
                min_tags=config.models.min_tags,
            ),
            image_fetcher=HttpImageFetcher(requests.Session(), request_policy),
            logger=logger,
            get_now=get_now,
            budget_minutes=config.models.extraction_budget_minutes,
        ),
        embedding_stage=EmbeddingStage(
            embedding_provider,
            logger,
            preload=lambda: load_tag_embeddings(db_path, config.models.embeddings),
        ),
        semantic_deduplicator=SemanticDeduplicationEngine(),
        similarity_stage=SimilarityStage(preferences, config.scoring),
        ranking_engine=RankingEngine(config, blocklist, logger),
        skipped_sources=skipped,
        candidate_repository=storage.candidates,
        event_repository=storage.events,
        run_repository=storage.runs,
        score_repository=storage.scores,
        ranking_repository=storage.rankings,
        dedup_decision_repository=storage.dedup_decisions,
        curve_state_repository=storage.curve_state,
        extraction_observation_repository=storage.extraction_observations,
        preference_revision_repository=storage.preference_revisions,
        preference_revision=preference_revision,
    )


def _with_accepted_curve(
    config: AppConfig, curve_state: CurveStateRepository, logger: Any
) -> AppConfig:
    """Apply the curve the last refit accepted, if there is one.

    Absent means the config defaults stand — a fresh deployment, or a regime
    that has never armed. That is the normal state and is not announced as a
    problem; the values in force are logged either way so a run's scores can be
    read against them.
    """
    state = curve_state.load()
    if state is None:
        logger.info(
            f"tag-confidence curve: config defaults "
            f"(cap {config.scoring.tag_confidence_cap}, "
            f"saturation {config.scoring.tag_confidence_saturation_chars})",
            component="scoring",
            duration_ms=0,
        )
        return config

    logger.info(
        f"tag-confidence curve: cap {state.cap:.3f}, saturation "
        f"{state.saturation:.1f} (regime {state.regime}, fitted {state.updated_at:%Y-%m-%d})",
        component="scoring",
        duration_ms=0,
    )
    return dataclasses.replace(
        config,
        scoring=dataclasses.replace(
            config.scoring,
            tag_confidence_cap=state.cap,
            tag_confidence_saturation_chars=state.saturation,
        ),
    )


def _handles(entities: EntityRepository, seeds_path: Path) -> list[str]:
    """Union the seed handles with every handle discovery has promoted.

    Seeds alone would make venue discovery inert — nothing would ever fetch
    what it promoted. The database alone would starve a first run, because
    `IngestionService` syncs seeds into the table only once it is already
    running, which is after these adapters are built.

    A handle promoted *during* tonight's run is therefore fetched from
    tomorrow, which is the right way round: it has not been vetted yet.
    """
    seeds = load_seeds(seeds_path) if seeds_path.exists() else None
    seed_handles = list(seeds.handles) if seeds is not None else []
    return sorted(
        set(seed_handles) | set(entities.active_handles())
    )



