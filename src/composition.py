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

from src.config import AppConfig
from src.enrichment.air_quality import OpenMeteoAirQualityProvider
from src.enrichment.astronomical import AstronomicalCalculator
from src.enrichment.movies import TMDbProvider
from src.enrichment.service import EnrichmentService
from src.enrichment.weather import OpenMeteoProvider
from src.ingestion.aggregators.do617_source import Do617VenueSource
from src.ingestion.aggregators.jsonld_source import JsonLdEventSource
from src.ingestion.calendars.html_source import HtmlListingSource
from src.ingestion.calendars.ics_source import IcsCalendarSource
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
from src.scoring.preferences import PreferenceRepository
from src.scoring.ranking import RankingEngine
from src.scoring.similarity_stage import SimilarityStage
from src.storage.entities import load_active_handles
from src.utils.logging import StructuredLogger
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

    handles = _handles(db_path, seeds_path)
    blocklist = _blocklist(blocklist_path)

    # Alternative routes to the same Instagram data, tried in order. Apify is
    # first because it is the only one under contract; the scrapers are the
    # fallback when it is unavailable or unaffordable.
    failover_sources: list[Any] = []
    apify_key = _credential("APIFY_API_KEY", "apify")
    if apify_key:
        failover_sources.append(ApifyAdapter(apify_key, handles, get_now=get_now))
    failover_sources.append(PicukiAdapter(handles, get_now=get_now))
    failover_sources.append(DumporAdapter(handles, get_now=get_now))

    independent_sources: list[Any] = []
    amc_key = _credential("AMC_API_KEY", "amc")
    if amc_key:
        independent_sources.append(
            AmcAdapter(amc_key, config.location.postal_code, get_now=get_now)
        )
    for feed in config.sources.ics_calendars:
        independent_sources.append(
            IcsCalendarSource(
                feed,
                db_path,
                get_now=get_now,
                logger=logger,
                timezone_name=config.location.timezone,
                horizon_days=config.scraping.horizon_days,
                day_starts_at=config.day_starts_at,
            )
        )
    for feed in config.sources.veezi_cinemas:
        independent_sources.append(
            VeeziSessionsSource(
                feed,
                db_path,
                get_now=get_now,
                logger=logger,
                timezone_name=config.location.timezone,
            )
        )
    for feed in config.sources.tribe_calendars:
        independent_sources.append(
            TribeCalendarSource(
                feed,
                db_path,
                get_now=get_now,
                logger=logger,
                timezone_name=config.location.timezone,
                horizon_days=config.scraping.horizon_days,
                day_starts_at=config.day_starts_at,
            )
        )
    for feed in config.sources.cabot_listings:
        independent_sources.append(
            CabotListingSource(
                feed,
                db_path,
                get_now=get_now,
                logger=logger,
                timezone_name=config.location.timezone,
                horizon_days=config.scraping.horizon_days,
                day_starts_at=config.day_starts_at,
            )
        )
    for feed in config.sources.do617_venues:
        independent_sources.append(
            Do617VenueSource(
                feed,
                db_path,
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
                db_path,
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
                db_path,
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
                db_path,
                config.location.timezone,
                get_now=get_now,
                logger=logger,
            )
        )

    ollama = OllamaClient(config.ollama_host)
    embedding_provider = OllamaEmbeddingProvider(ollama, model=config.models.embeddings)

    tmdb_key = _credential("TMDB_API_KEY", "tmdb")

    preferences = PreferenceRepository(embedding_provider, db_path, logger).load(
        likes_path, dislikes_path
    )

    return BatchDependencies(
        ingestion_service=IngestionService(
            config=config,
            db_path=db_path,
            seeds_path=seeds_path,
            failover_sources=failover_sources,
            independent_sources=independent_sources,
            logger=logger,
            blocklist=blocklist,
        ),
        normalization_service=NormalizationService(config, logger),
        enrichment_service=EnrichmentService(
            weather_provider=OpenMeteoProvider(),
            movie_provider=TMDbProvider(tmdb_key) if tmdb_key else None,
            astronomical_calculator=AstronomicalCalculator(),
            synthetic_rules=config.synthetic_activities,
            config=config,
            db_path=db_path,
            air_quality_provider=OpenMeteoAirQualityProvider(),
            get_now=get_now,
            logger=logger,
        ),
        extraction_stage=ExtractionStage(
            provider=OllamaExtractionProvider(
                ollama,
                model=config.models.llm_extraction,
                min_tags=config.scoring.min_tags_per_event,
            ),
            image_fetcher=HttpImageFetcher(),
            logger=logger,
            get_now=get_now,
        ),
        embedding_stage=EmbeddingStage(embedding_provider, logger),
        semantic_deduplicator=SemanticDeduplicationEngine(),
        similarity_stage=SimilarityStage(preferences, config.scoring),
        ranking_engine=RankingEngine(config, blocklist, logger),
        skipped_sources=skipped,
    )


def _handles(db_path: Path, seeds_path: Path) -> list[str]:
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
    return sorted(set(seed_handles) | set(load_active_handles(db_path)))


def _blocklist(path: Path) -> list[str]:
    """Read `blocklist.json`, treating an absent file as an empty blocklist.

    A user who never wrote one is a normal deployment, not a failure. The file
    stays the source of truth — see #16 for populating the table from it.
    """
    if not path.exists():
        return []
    entries = json.loads(path.read_text())
    return [str(entry) for entry in entries]
