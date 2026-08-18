"""The one place a concrete storage implementation is named.

Two processes need storage and they need different amounts of it: the batch
persists through every repository, the CLI reads through three. They are
separate composition roots because a root is per-*process*, not per-application
— having the query path import the batch root would drag in the LLM client it is
architecturally forbidden to use, measured at 56 extra modules.

What they must not do is wire storage *independently*, which is what they were
doing. The batch passed `config.models.embeddings` to the event repository and
the CLI took the default; identical only for as long as `config.yaml` names the
model that happens to be `DEFAULT_EMBEDDING_MODEL`. Change that line and the
batch writes vectors under the new name while the CLI reads under the old one,
finding none, silently.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.storage.protocols import (
    CurveStateRepository,
    ExtractionObservationRepository,
    CandidateRepository,
    EntityRepository,
    EventRepository,
    HttpCache,
    MovieCache,
    PreferenceRevisionRepository,
    RankingRepository,
    RescoreRepository,
    RunRepository,
    DedupDecisionRepository,
    ScoreRepository,
    DayCache,
)
from src.storage.sqlite.candidates import SqliteCandidateRepository
from src.storage.sqlite.entities import SqliteEntityRepository
from src.storage.sqlite.events import SqliteEventRepository
from src.storage.sqlite.http_cache import SqliteHttpCache
from src.storage.sqlite.rankings import SqliteRankingRepository
from src.storage.sqlite.curve_state import SqliteCurveStateRepository
from src.storage.sqlite.extraction_observations import (
    SqliteExtractionObservationRepository,
)
from src.storage.sqlite.preference_revisions import (
    SqlitePreferenceRevisionRepository,
)
from src.storage.sqlite.rescores import SqliteRescoreRepository
from src.storage.sqlite.runs import SqliteRunRepository
from src.storage.sqlite.dedup_decisions import SqliteDedupDecisionRepository
from src.storage.sqlite.scores import SqliteScoreRepository
from src.storage.sqlite.day_cache import SqliteDayCache
from src.storage.sqlite.movie_cache import SqliteMovieCache


@dataclass(frozen=True)
class ViewStorage:
    """What the CLI reads and, on a rescore, writes back through.

    It was read-only until the read path gained the ability to recompute a
    stale ranking. That is still the exception rather than the rule: the view
    writes only when it has just recomputed what it is about to show.
    """

    events: EventRepository
    scores: ScoreRepository
    rankings: RankingRepository
    #: Read to notice that the preference files have moved since the ranking
    #: was scored.
    preference_revisions: PreferenceRevisionRepository
    #: Written, unlike everything above. A read-time rescore replaces a
    #: stored ordering, and this is what records that it did.
    rescores: RescoreRepository
    #: Read only for `open_run`: a batch in flight is the one condition
    #: under which the read path must not write at all.
    runs: RunRepository
    #: The rescore's forecast, served from cache when it is still fresh —
    #: which is what makes a second invocation seconds later open no socket.
    weather_cache: DayCache
    #: Its own table, not the forecast's. Both are keyed
    #: `(date, latitude, longitude)`, so one table would collide.
    air_quality_cache: DayCache


@dataclass(frozen=True)
class BatchStorage:
    """What the batch persists through."""

    events: EventRepository
    scores: ScoreRepository
    rankings: RankingRepository
    candidates: CandidateRepository
    runs: RunRepository
    entities: EntityRepository
    dedup_decisions: DedupDecisionRepository
    weather_cache: DayCache
    air_quality_cache: DayCache
    movie_cache: MovieCache
    http_cache: HttpCache
    curve_state: CurveStateRepository
    extraction_observations: ExtractionObservationRepository
    preference_revisions: PreferenceRevisionRepository


def build_view_storage(db_path: Path | str, embedding_model: str) -> ViewStorage:
    """The three stores the CLI joins into a listing.

    Args:
        db_path: Database to read.
        embedding_model: Which model's vectors to attach. Must match what the
            batch wrote, which is why it is a parameter rather than a default.
    """
    return ViewStorage(
        events=SqliteEventRepository(db_path, embedding_model),
        scores=SqliteScoreRepository(db_path),
        rankings=SqliteRankingRepository(db_path),
        preference_revisions=SqlitePreferenceRevisionRepository(db_path),
        rescores=SqliteRescoreRepository(db_path),
        runs=SqliteRunRepository(db_path),
        weather_cache=SqliteDayCache(db_path, table="weather_cache"),
        air_quality_cache=SqliteDayCache(db_path, table="air_quality_cache"),
    )


def build_batch_storage(db_path: Path | str, embedding_model: str) -> BatchStorage:
    """Everything a batch reads and writes.

    Args:
        db_path: Database to read and write.
        embedding_model: Names which model produced the vectors, so a tag
            embedded by another model is left unattached rather than silently
            mixed into the same space.
    """
    return BatchStorage(
        events=SqliteEventRepository(db_path, embedding_model),
        scores=SqliteScoreRepository(db_path),
        rankings=SqliteRankingRepository(db_path),
        candidates=SqliteCandidateRepository(db_path),
        runs=SqliteRunRepository(db_path),
        curve_state=SqliteCurveStateRepository(db_path),
        extraction_observations=SqliteExtractionObservationRepository(db_path),
        preference_revisions=SqlitePreferenceRevisionRepository(db_path),
        entities=SqliteEntityRepository(db_path),
        dedup_decisions=SqliteDedupDecisionRepository(db_path),
        weather_cache=SqliteDayCache(db_path, table="weather_cache"),
        air_quality_cache=SqliteDayCache(db_path, table="air_quality_cache"),
        movie_cache=SqliteMovieCache(db_path),
        http_cache=SqliteHttpCache(db_path),
    )
