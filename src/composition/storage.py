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
    CandidateRepository,
    EntityRepository,
    EventRepository,
    HttpCache,
    RankingRepository,
    RunRepository,
    ScoreRepository,
    WeatherCache,
)
from src.storage.sqlite.candidates import SqliteCandidateRepository
from src.storage.sqlite.entities import SqliteEntityRepository
from src.storage.sqlite.events import SqliteEventRepository
from src.storage.sqlite.http_cache import SqliteHttpCache
from src.storage.sqlite.rankings import SqliteRankingRepository
from src.storage.sqlite.runs import SqliteRunRepository
from src.storage.sqlite.scores import SqliteScoreRepository
from src.storage.sqlite.weather_cache import SqliteWeatherCache


@dataclass(frozen=True)
class ViewStorage:
    """What the CLI reads through. No writer, because the view never writes."""

    events: EventRepository
    scores: ScoreRepository
    rankings: RankingRepository


@dataclass(frozen=True)
class BatchStorage:
    """What the batch persists through."""

    events: EventRepository
    scores: ScoreRepository
    rankings: RankingRepository
    candidates: CandidateRepository
    runs: RunRepository
    entities: EntityRepository
    weather_cache: WeatherCache
    http_cache: HttpCache


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
        entities=SqliteEntityRepository(db_path),
        weather_cache=SqliteWeatherCache(db_path),
        http_cache=SqliteHttpCache(db_path),
    )
