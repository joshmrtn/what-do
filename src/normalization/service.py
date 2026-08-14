"""NormalizationService — orchestrates normalization and dedup pass 1."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable

from src.config import AppConfig
from src.models.event import Event
from src.models.event_candidate import EventCandidate
from src.normalization.deduplicator import DeduplicationEngine, MergeDecision
from src.normalization.normalizer import NormalizationEngine
from src.utils.logging import StructuredLogger


@dataclass
class NormalizationResult:
    """Result returned by NormalizationService.run()."""

    normalized: int
    discarded: int
    events: list[Event]
    #: Every comparison dedup Pass 1 made. Carried rather than written here:
    #: this service does not persist, and the orchestrator owns the run id
    #: these rows reference.
    decisions: list[MergeDecision] = field(default_factory=list)


class NormalizationService:
    """Orchestrate normalization → dedup pass 1 for a batch of candidates.

    Follows the same pattern as IngestionService: constructed with config and
    dependencies, driven via run().

    It does not persist. The events it returns still carry the normalizer's
    throwaway uuids, and the batch orchestrator reconciles those against stored
    identity before anything is written.
    """

    def __init__(
        self,
        config: AppConfig,
        logger: StructuredLogger,
    ) -> None:
        """
        Args:
            config: Application config (timezone, dedup thresholds).
            logger: Structured logger for discard events.
        """
        self._config = config
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
            NormalizationResult with the deduplicated events and their counts.
        """
        norm_result = self._normalizer.normalize(candidates, get_now=get_now)

        for discard in norm_result.discards:
            self._logger.warning(
                f"EventCandidate discarded from {discard.candidate.source}: "
                f"{discard.reason}",
                component="normalization",
                duration_ms=0,
            )

        deduped = self._deduplicator.deduplicate(
            norm_result.events, self._config.deduplication
        )
        events = deduped.events

        return NormalizationResult(
            decisions=deduped.decisions,
            normalized=len(events),
            discarded=len(norm_result.discards),
            events=events,
        )
