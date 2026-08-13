"""Deterministic ranking — turns scored events into one run's ordered recommendations.

The ordering is the product. Everything here is reproducible from its inputs:
no clock, no randomness, no network, and ties broken explicitly so upstream
iteration order can never leak into the result.

Nothing is excluded. Every event ranks, including deeply negative ones — the
order is the whole product, and withholding an event would hide a judgement
rather than express one. The single exception is a blocklisted venue, which is
user intent rather than a scoring judgement.
"""

from __future__ import annotations

import math

from dataclasses import replace
from datetime import date

from src.config import AppConfig
from src.models.event import Event
from src.processing.extraction_input import extraction_input
from src.models.event_score import EventScore
from src.models.ranking import Ranking
from src.scoring.similarity import Reason, SimilarityResult
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

    def rank(
        self, events: list[Event], run_date: date
    ) -> tuple[list[EventScore], list[Ranking]]:
        """Score and order every event for one batch run.

        The two halves are returned apart because they are stored apart and
        answer different questions: a score is the verdict on an event under
        the current preferences, a ranking is its place in one night's order.

        Args:
            events: Scored events, each carrying a `similarity` result.
            run_date: The batch date to stamp on both halves.

        Returns:
            Every event's score, and a placement for each, ordered best first
            and numbered 1..N. Scores are returned in the same order as the
            placements so the two can be zipped.
        """
        pairs = [
            self._score(event, run_date)
            for event in events
            if not self._drop_as_blocked(event)
        ]

        ordered = sorted(pairs, key=lambda pair: (-pair[1], pair[0].event_id))

        scores = [score for score, _, _ in ordered]
        rankings = [
            Ranking(
                event_id=score.event_id,
                run_date=run_date,
                weather_adjustment=adjustment,
                final_score=final_score,
                rank=position,
            )
            for position, (score, final_score, adjustment) in enumerate(ordered, start=1)
        ]
        return scores, rankings

    def _score(self, event: Event, run_date: date) -> tuple[EventScore, float, float]:
        """One event's score, its final score and its weather adjustment.

        The latter two are not on `EventScore` — they belong to the placement —
        but ordering needs them before a `Ranking` can be built, so they ride
        along until `rank` is known.
        """
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
                        f"{len(event.tags)} tags from "
                        f"{len(extraction_input(event))} characters"
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

        score = EventScore(
            event_id=event.event_id,
            run_date=run_date,
            tag_score=similarity.tag_score,
            summary_score=similarity.summary_score,
            base_score=base_score,
            tag_confidence=confidence,
            match=similarity.match,
            reasons=reasons,
        )
        return score, final_score, adjustment

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

        expected = self._expected_tags(event)
        if expected <= 0:
            return 1.0
        return min(1.0, len(event.tags) / expected)

    def _expected_tags(self, event: Event) -> float:
        """How many tags this event's input could reasonably have earned.

        `cap × (1 − e^(−chars / saturation))`, over the same text extraction
        read. A fixed floor asked the same question of a 25-character cinema
        listing and a 2,000-character festival programme, so the terse source
        was marked down for being terse — punishing honesty rather than
        detecting a thin extraction.

        Deliberately a crude fit. Length is a weak proxy for how much a
        description actually distinguishes; a long one can honestly earn one
        tag. It only has to beat a constant.
        """
        scoring = self._config.scoring
        saturation = scoring.tag_confidence_saturation_chars
        if saturation <= 0:
            return float(scoring.tag_confidence_cap)
        chars = len(extraction_input(event))
        return scoring.tag_confidence_cap * (1.0 - math.exp(-chars / saturation))

    def _multiplier(self, match: str) -> float:
        scoring = self._config.scoring
        return {
            "yes": scoring.match_multiplier_yes,
            "maybe": scoring.match_multiplier_maybe,
            "no": scoring.match_multiplier_no,
        }.get(match, scoring.match_multiplier_maybe)

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
