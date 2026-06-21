"""NormalizationService — orchestrates normalization, dedup, and persistence."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

from src.config import AppConfig
from src.models.event import Event
from src.models.event_candidate import EventCandidate
from src.normalization.deduplicator import DeduplicationEngine
from src.normalization.normalizer import NormalizationEngine
from src.utils.logging import StructuredLogger


@dataclass
class NormalizationResult:
    """Result returned by NormalizationService.run()."""

    persisted: int
    discarded: int


def _persist_events(events: list[Event], db_path: Path) -> None:
    """Insert normalized events into the events table."""
    conn = sqlite3.connect(db_path)
    try:
        conn.executemany(
            """
            INSERT OR REPLACE INTO events (
                id, source_event_candidates, source_type,
                url, image_url, title, venue, description, location,
                start_time, end_time, tags, summary,
                tag_embeddings, summary_embedding,
                weather, astronomical_data, metadata,
                created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [_event_to_row(e) for e in events],
        )
        conn.commit()
    finally:
        conn.close()


def _event_to_row(e: Event) -> tuple:
    return (
        e.event_id,
        json.dumps(e.source_event_candidates),
        e.source_type,
        e.url,
        e.image_url,
        e.title,
        e.venue,
        e.description,
        e.location,
        e.start_time.isoformat() if e.start_time else None,
        e.end_time.isoformat() if e.end_time else None,
        json.dumps(e.tags),
        e.summary,
        None,  # tag_embeddings — populated in Phase 7
        e.summary_embedding,
        json.dumps(e.weather) if e.weather else None,
        json.dumps(e.astronomical_data) if e.astronomical_data else None,
        json.dumps(e.metadata),
        e.created_at.isoformat(),
        e.updated_at.isoformat(),
    )


class NormalizationService:
    """Orchestrate normalization → dedup → persistence for a batch of candidates.

    Follows the same pattern as IngestionService: constructed with config and
    dependencies, driven via run().
    """

    def __init__(
        self,
        config: AppConfig,
        db_path: Path,
        logger: StructuredLogger,
    ) -> None:
        """
        Args:
            config: Application config (timezone, dedup thresholds).
            db_path: Path to SQLite database.
            logger: Structured logger for discard events.
        """
        self._config = config
        self._db_path = Path(db_path)
        self._logger = logger
        self._normalizer = NormalizationEngine(
            timezone_name=config.location.timezone
        )
        self._deduplicator = DeduplicationEngine()

    def run(
        self,
        candidates: list[EventCandidate],
        get_now: Callable[[], datetime] = datetime.now,
    ) -> NormalizationResult:
        """Normalize, deduplicate, and persist a list of EventCandidates.

        Args:
            candidates: Raw candidates from the ingestion layer.
            get_now: Injectable clock for event timestamps.

        Returns:
            NormalizationResult with persisted and discard counts.
        """
        norm_result = self._normalizer.normalize(candidates, get_now=get_now)

        for discard in norm_result.discards:
            self._logger.warning(
                f"EventCandidate discarded from {discard.candidate.source}: "
                f"{discard.reason}",
                component="normalization",
                duration_ms=0,
            )

        events = self._deduplicator.deduplicate(
            norm_result.events, self._config.deduplication
        )

        if events:
            _persist_events(events, self._db_path)

        return NormalizationResult(
            persisted=len(events),
            discarded=len(norm_result.discards),
        )
