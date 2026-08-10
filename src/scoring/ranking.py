"""Deterministic ranking — turns scored events into one run's ordered recommendations.

The ordering is the product. Everything here is reproducible from its inputs:
no clock, no randomness, no network, and ties broken explicitly so upstream
iteration order can never leak into the result.

Nothing is excluded. Every event ranks, including deeply negative ones, because
`tier` is a label the CLI renders and never a filter. The single exception is a
blocklisted venue, which is user intent rather than a scoring judgement.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import date

from src.config import AppConfig
from src.models.event import Event
from src.models.recommendation import Recommendation, make_recommendation_id
from src.scoring.similarity import Reason, SimilarityResult
from src.scoring.tiers import (
    EVERYTHING_ELSE,
    TOP_PICK,
    WORTH_CONSIDERING,
    tier_for_score,
)
from src.scoring.weather_score import weather_adjustment
from src.utils.blocklist import is_blocked
from src.utils.logging import StructuredLogger, get_logger


MATCH_FACTOR = "match_classification"
CONFIDENCE_FACTOR = "low_tag_confidence"

_UNSCORED = SimilarityResult()


class RankingEngine:
    """Ranks scored events into a run's recommendations.

    Pure given its inputs — `run_date` is passed in rather than read from a
    clock, and the blocklist is supplied rather than loaded, so two runs over
    the same events produce identical output.

    Args:
        config: Scoring thresholds and multipliers, comfort curves, and the
            blocklist match threshold.
        blocklist: Raw entries from `data/blocklist.json`. Only venue-name
            entries apply here; `@handle` entries are enforced at ingestion,
            where the handle still exists on the candidate.
        logger: Injected logger. Blocklist drops are logged so a too-aggressive
            fuzzy match is visible rather than silent.
    """

    def __init__(
        self,
        config: AppConfig,
        blocklist: list[str] | None = None,
        logger: StructuredLogger | None = None,
    ) -> None:
        self._config = config
        self._blocklist = blocklist or []
        self._logger = logger or get_logger("ranking")

    def rank(self, events: list[Event], run_date: date) -> list[Recommendation]:
        """Score, order, and label every event for one batch run.

        Args:
            events: Scored events, each carrying a `similarity` result.
            run_date: The batch date to stamp on every recommendation.

        Returns:
            Recommendations ordered best first, ranked 1..N.
        """
        scored = [
            self._score(event, run_date)
            for event in events
            if not self._drop_as_blocked(event)
        ]

        ordered = sorted(scored, key=lambda r: (-r.final_score, r.event_id))

        return [
            replace(recommendation, rank=position)
            for position, recommendation in enumerate(ordered, start=1)
        ]

    def _score(self, event: Event, run_date: date) -> Recommendation:
        """Build one event's recommendation. `rank` is filled in once ordered."""
        similarity = event.similarity or _UNSCORED
        base_score = similarity.base_score
        reasons = list(similarity.reasons)

        confidence = self._tag_confidence(event)
        confident = base_score * confidence
        if confidence < 1.0:
            reasons.append(
                Reason(
                    factor=CONFIDENCE_FACTOR,
                    matched_preference=(
                        f"{len(event.tags)} of {self._config.scoring.min_tags_per_event} tags"
                    ),
                    similarity=confidence,
                    contribution=confident - base_score,
                    direction="positive" if confident >= base_score else "negative",
                    tag=None,
                )
            )

        multiplier = self._multiplier(similarity.match)
        # Direction-aware: the multiplier acts on magnitude, never on sign. A
        # plain product would turn `no` at 0.5 into a *better* score whenever
        # the base is negative, rewarding the clearest rejections.
        adjusted = confident * multiplier if confident >= 0 else confident / multiplier
        reasons.append(
            Reason(
                factor=MATCH_FACTOR,
                matched_preference=similarity.match,
                similarity=multiplier,
                contribution=adjusted - confident,
                direction="positive" if adjusted >= confident else "negative",
                tag=None,
            )
        )

        adjustment, weather_reason = weather_adjustment(event, self._config.weather)
        if weather_reason is not None:
            reasons.append(weather_reason)

        final_score = adjusted + adjustment

        return Recommendation(
            recommendation_id=make_recommendation_id(run_date, event.event_id),
            event_id=event.event_id,
            run_date=run_date,
            base_score=base_score,
            weather_adjustment=adjustment,
            tag_confidence=confidence,
            final_score=final_score,
            match=similarity.match,
            tier=self._tier(final_score),
            rank=0,
            reasons=reasons,
        )

    def _tag_confidence(self, event: Event) -> float:
        """How much of the expected tag count the extraction actually produced.

        Applied symmetrically, unlike the match multiplier: it shrinks a score
        toward zero in both directions, because a thin tag list means the
        evidence is weak, not that the verdict is bad. An event we know almost
        nothing about belongs in the middle of the ranking.
        """
        # Tags authored in config, so a low count is an authoring choice rather
        # than a failed extraction.
        if event.is_synthetic:
            return 1.0

        expected = self._config.scoring.min_tags_per_event
        if expected <= 0:
            return 1.0
        return min(1.0, len(event.tags) / expected)

    def _multiplier(self, match: str) -> float:
        scoring = self._config.scoring
        return {
            "yes": scoring.match_multiplier_yes,
            "maybe": scoring.match_multiplier_maybe,
            "no": scoring.match_multiplier_no,
        }.get(match, scoring.match_multiplier_maybe)

    def _tier(self, final_score: float) -> str:
        """Label the score. Presentation only — never a filter."""
        scoring = self._config.scoring
        return tier_for_score(
            final_score, scoring.top_picks_min, scoring.worth_considering_min
        )

    def _drop_as_blocked(self, event: Event) -> bool:
        """Whether this event's venue is blocklisted.

        The last line of defence: discovery and ingestion both filter earlier,
        but an event persisted before its venue was blocked would otherwise be
        recommended forever.
        """
        blocked = is_blocked(
            event.venue,
            [],
            self._blocklist,
            self._config.venue_discovery.blocklist_name_match_threshold,
        )
        if blocked:
            self._logger.info(
                f"Dropping blocklisted venue from ranking: {event.venue}",
                component="ranking",
                duration_ms=0,
            )
        return blocked
