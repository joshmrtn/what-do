"""Composition root for the read path's rescore.

Separate from `batch.py`, and it must stay separate: importing the batch root
drags in every adapter and the Ollama client, measured at 41 → 105 modules on
the one import path whose promise is no model call at query time.

What it builds is the batch's tail with **extraction** absent — and only
extraction. There is no extraction stage here at all: not a disabled one, not one
behind a flag. Minutes an event is the only cost that cannot be paid at query
time, and it is the whole of the rule.

Embeddings are built for real and are **meant** to run. An edited `likes.txt`
introduces preference lines with no cached vector; embedding them costs about a
second each, and seeing how they reshuffle tonight without waiting for the
overnight batch is the reason this path exists. `CLAUDE.md` forbids **LLM** calls
at query time and nothing else; reading that as a ban on embeddings was an
over-tightening, corrected 2026-08-17, and the rule now says so in as many words.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

from src.composition.pipeline import load_blocklist
from src.composition.storage import ViewStorage
from src.config import (
    DEFAULT_BLOCKLIST_PATH,
    DEFAULT_DISLIKES_PATH,
    DEFAULT_LIKES_PATH,
    AppConfig,
)
from src.enrichment.air_quality import OpenMeteoAirQualityProvider
from src.enrichment.astronomical import AstronomicalCalculator
from src.enrichment.service import EnrichmentService
from src.enrichment.weather import WeatherProvider
from src.scoring.embedding_stage import EmbeddingStage
from src.scoring.embeddings import OllamaEmbeddingProvider
from src.models.preference_revision import PreferenceRevision
from src.scoring.preference_revision import build_revision
from src.scoring.preferences import PreferenceRepository
from src.scoring.ranking import RankingEngine
from src.scoring.similarity_stage import SimilarityStage
from src.storage.events import load_tag_embeddings
from src.utils.ollama_client import OllamaClient
from src.utils.logging import StructuredLogger


@dataclass(frozen=True)
class RescorePipeline:
    """The stages a read-time rescore runs, already configured."""

    enrichment_service: EnrichmentService
    embedding_stage: EmbeddingStage
    similarity_stage: SimilarityStage
    ranking_engine: RankingEngine
    #: What the preference files said when this rescore loaded them. Carried
    #: rather than looked up afterwards: `latest()` answers "what did the last
    #: batch use", which is a different question and is None until one has run.
    preference_revision: PreferenceRevision


def build_rescore_pipeline(
    *,
    config: AppConfig,
    db_path: Path,
    logger: StructuredLogger,
    get_now: Callable[[], datetime],
    storage: ViewStorage,
    weather_provider: WeatherProvider,
    likes_path: Path = DEFAULT_LIKES_PATH,
    dislikes_path: Path = DEFAULT_DISLIKES_PATH,
    blocklist_path: Path = DEFAULT_BLOCKLIST_PATH,
) -> RescorePipeline:
    """Build the tail the read path re-runs over a stale ranking.

    Args:
        config: Loaded application config. The comfort curves, the scoring
            constants and the zone all come from here, so a rescore and the
            batch that preceded it agree by construction.
        db_path: Database, for the weather cache and the preference vectors.
        logger: Structured logger.
        get_now: Injectable clock.
        storage: The view's repositories, so the rescore writes back through the
            same objects the listing was read through.
        weather_provider: Where the fresh forecast comes from. Injected
            because it is the one genuine external boundary on this path,
            and a test of the rescore should not need a socket to have one.
        likes_path: Preference file. A parameter rather than a constant read
            inside, so this is reachable from a test without owning two files at
            fixed locations.
        dislikes_path: The same, for dislikes.
        blocklist_path: Venue names never to surface.

    Returns:
        The configured stages.

    Raises:
        EmbeddingError: If a preference line cannot be embedded at all — Ollama
            unreachable, rather than a line simply being new. The caller treats
            this as "fall back to the stored ranking", never as a reason to lose
            the listing.
    """
    # Built exactly as the batch builds it, so a vector produced here and one
    # produced overnight are bit-identical. No chat parameters: `/api/embed`
    # takes none and the embedding model neither samples nor reasons.
    embedding_provider = OllamaEmbeddingProvider(
        OllamaClient(
            config.ollama_host,
            timeout=config.models.request_timeout_seconds,
            component="embedding",
            keep_alive=config.models.keep_alive,
        ),
        model=config.models.embeddings,
    )

    # No movie provider. TMDb has no cache yet, so building one here would put
    # a third-party call on the read path — and the ordering that keeps this
    # honest is already recorded: the cache lands before the key is uncommented,
    # and this root then picks the provider up on the same terms as the batch.
    #
    # This is where an edited preference file is paid for: a changed line has no
    # cached vector, so loading embeds it. About a second a line, which is the
    # point.
    preferences = PreferenceRepository(embedding_provider, db_path, logger).load(
        likes_path, dislikes_path
    )

    return RescorePipeline(
        enrichment_service=EnrichmentService(
            weather_provider=weather_provider,
            movie_provider=None,
            astronomical_calculator=AstronomicalCalculator(),
            synthetic_rules=config.synthetic_activities,
            config=config,
            db_path=db_path,
            weather_cache=storage.weather_cache,
            air_quality_provider=OpenMeteoAirQualityProvider(),
            get_now=get_now,
            logger=logger,
        ),
        # Ordinarily a no-op: tags change only through extraction, which is not
        # on this path, so `embedding_input_hash` is invariant and every event
        # hits the skip. Where it is not a no-op it does the work rather than
        # refusing — an event the batch left unembedded is embedded here.
        embedding_stage=EmbeddingStage(
            embedding_provider,
            logger,
            # The batch passes this and so must the view. A vector is a pure
            # function of its text and the model, so a regenerated synthetic
            # activity's authored tags were embedded on some previous night
            # and need no model now — without the memo the refusing provider
            # fires and every rescore is abandoned the moment a synthetic
            # activity qualifies.
            preload=lambda: load_tag_embeddings(db_path, config.models.embeddings),
        ),
        similarity_stage=SimilarityStage(preferences, config.scoring),
        ranking_engine=RankingEngine(
            config, load_blocklist(blocklist_path), logger
        ),
        preference_revision=build_revision(
            preferences,
            likes_name=likes_path.name,
            dislikes_name=dislikes_path.name,
            captured_at=get_now(),
        ),
    )
