"""NormalizationService — orchestrates normalization, dedup, and persistence."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

from src.config import AppConfig
from src.models.event import Event
from src.models.event_candidate import EventCandidate
from src.normalization.deduplicator import DeduplicationEngine
from src.normalization.normalizer import NormalizationEngine
from src.storage.events import save_events
from src.utils.logging import StructuredLogger


@dataclass
class NormalizationResult:
    """Result returned by NormalizationService.run()."""

    persisted: int
    discarded: int


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
            save_events(events, self._db_path)

        return NormalizationResult(
            persisted=len(events),
            discarded=len(norm_result.discards),
        )
