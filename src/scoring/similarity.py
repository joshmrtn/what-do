"""Similarity engine — turns embeddings into an explainable, deterministic score.

The formula here replaces the one originally specified in the high-level design,
which was measured against real vectors and found to score every candidate venue
negative. See `docs/decisions.md` — "Scoring formula replaced after measurement".

Four elements are load-bearing; dropping any one breaks real cases:

1. A logistic **gate** on each similarity, so noise cannot vote. Unrelated pairs
   sit around 0.40, and the raw rule handed a coin-flip between two noise values
   a near-full-magnitude contribution.
2. Tag **weights multiply** the contribution. Used as averaging weights they
   normalise away and suppress nothing.
3. **balanced_mean** aggregation, so several weak incidental negatives cannot
   outvote one strong positive.
4. A **relative margin** for the `no` label. An absolute dislike cutoff rejects
   a karaoke bar, where "bar" scores 0.932 against the dislike "bars".
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from src.config import GENERAL_DOMAIN_DEFAULT, ScoringConfig
from src.models.event import Event
from src.scoring.preferences import PreferenceSet, UserPreference
from src.utils.vectors import cosine, decode_vector

LIKE_FACTOR = "like_similarity"
DISLIKE_FACTOR = "dislike_similarity"


@dataclass(frozen=True)
class Reason:
    """One contribution to an event's score, in human-auditable form.

    Args:
        factor: Which scoring factor produced this contribution.
        matched_preference: The preference line that matched most closely.
        similarity: Raw cosine similarity to that preference, before gating.
        contribution: Signed contribution after gating and weighting.
        direction: "positive" or "negative".
        tag: The tag this came from, or None for the summary term.
    """

    factor: str
    matched_preference: str
    similarity: float
    contribution: float
    direction: str
    tag: str | None = None


@dataclass
class SimilarityResult:
    """Scored output for one event, consumed by the ranking engine."""

    tag_score: float = 0.0
    summary_score: float = 0.0
    base_score: float = 0.0
    match: str = "maybe"
    reasons: list[Reason] = field(default_factory=list)


def _gate(similarity: float, midpoint: float, temperature: float) -> float:
    """Logistic ramp: ~1 well above the midpoint, decaying smoothly to ~0 below.

    Maps the embedding noise band to almost exactly zero while letting a
    near-miss still count faintly, which a hard threshold cannot do.
    """
    if temperature <= 0:
        return 1.0 if similarity >= midpoint else 0.0
    return 1.0 / (1.0 + math.exp(-(similarity - midpoint) / temperature))


def _best(vector: list[float], prefs: list[UserPreference]) -> tuple[UserPreference | None, float]:
    """Closest preference to the vector, and its similarity."""
    best_pref: UserPreference | None = None
    best_sim = 0.0
    for pref in prefs:
        if not pref.embedding:
            continue
        similarity = cosine(vector, pref.embedding)
        if best_pref is None or similarity > best_sim:
            best_pref, best_sim = pref, similarity
    return best_pref, best_sim


def _contribution(
    vector: list[float],
    weight: float,
    likes: list[UserPreference],
    dislikes: list[UserPreference],
    cfg: ScoringConfig,
    tag: str | None,
) -> tuple[Reason | None, float, float]:
    """Score one vector, returning (reason, best_like_sim, best_dislike_sim)."""
    like_pref, like_sim = _best(vector, likes)
    dislike_pref, dislike_sim = _best(vector, dislikes)

    if like_pref is None and dislike_pref is None:
        return None, 0.0, 0.0

    # Specificity wins: the closer match sets the direction.
    if like_pref is not None and like_sim > dislike_sim:
        winner, similarity, factor, sign = like_pref, like_sim, LIKE_FACTOR, 1.0
    elif dislike_pref is not None:
        winner, similarity, factor, sign = dislike_pref, dislike_sim, DISLIKE_FACTOR, -1.0
    else:
        return None, like_sim, dislike_sim

    contribution = (
        sign * similarity * _gate(similarity, cfg.gate_midpoint, cfg.gate_temperature) * weight
    )

    return (
        Reason(
            factor=factor,
            matched_preference=winner.text,
            similarity=similarity,
            contribution=contribution,
            direction="positive" if sign > 0 else "negative",
            tag=tag,
        ),
        like_sim,
        dislike_sim,
    )


def _balanced_mean(contributions: list[float]) -> float:
    """mean(positives) - mean(|negatives|).

    Compares the average strength of positive evidence against the average
    strength of negative evidence, so a count of weak incidental negatives
    cannot outvote one strong positive. An empty side contributes 0.
    """
    positives = [c for c in contributions if c > 0]
    negatives = [-c for c in contributions if c < 0]
    mean_pos = sum(positives) / len(positives) if positives else 0.0
    mean_neg = sum(negatives) / len(negatives) if negatives else 0.0
    return mean_pos - mean_neg


def _specificity_sum(contributions: list[float]) -> float:
    """sum(contributions) / count — the originally specified aggregator."""
    if not contributions:
        return 0.0
    return sum(contributions) / len(contributions)


_AGGREGATORS = {
    "balanced_mean": _balanced_mean,
    "specificity_sum": _specificity_sum,
}


class SimilarityEngine:
    """Scores an event against the user's preferences.

    Pure — no I/O, no DB access, no clock. Identical inputs always produce
    identical output, which the ranking engine depends on.
    """

    def score(
        self, event: Event, preferences: PreferenceSet, config: ScoringConfig
    ) -> SimilarityResult:
        """Score one event against domain-applicable preferences.

        Args:
            event: An embedded event.
            preferences: Loaded likes and dislikes.
            config: Scoring thresholds, gate shape, and domain mapping.

        Returns:
            SimilarityResult with scores, match label, and per-tag reasons.
        """
        domain = config.domain_map.get(event.source_type, GENERAL_DOMAIN_DEFAULT)
        likes = preferences.likes_for(domain)
        dislikes = preferences.dislikes_for(domain)

        reasons: list[Reason] = []
        contributions: list[float] = []
        best_like = 0.0
        best_dislike = 0.0

        for tag, blob in zip(event.tags, event.tag_embeddings):
            reason, like_sim, dislike_sim = _contribution(
                decode_vector(blob), tag.weight, likes, dislikes, config, tag.text
            )
            best_like = max(best_like, like_sim)
            best_dislike = max(best_dislike, dislike_sim)
            if reason is not None:
                reasons.append(reason)
                contributions.append(reason.contribution)

        aggregate = _AGGREGATORS[config.aggregator]
        tag_score = aggregate(contributions)

        summary_score = 0.0
        if event.summary_embedding is not None:
            summary_reason, like_sim, dislike_sim = _contribution(
                decode_vector(event.summary_embedding), 1.0, likes, dislikes, config, None
            )
            best_like = max(best_like, like_sim)
            best_dislike = max(best_dislike, dislike_sim)
            if summary_reason is not None:
                reasons.append(summary_reason)
                summary_score = summary_reason.contribution

        base_score = tag_score + config.summary_weight * summary_score

        return SimilarityResult(
            tag_score=tag_score,
            summary_score=summary_score,
            base_score=base_score,
            match=self._classify(base_score, best_like, best_dislike, config),
            reasons=reasons,
        )

    @staticmethod
    def _classify(
        base_score: float, best_like: float, best_dislike: float, cfg: ScoringConfig
    ) -> str:
        """Assign the advisory yes/maybe/no label.

        `no` is relative, never an absolute dislike cutoff: the strongest
        dislike must beat the strongest like by a margin. Measured, an absolute
        rule rejects a karaoke bar the user likes, where "bar" scores 0.932
        against "bars" but karaoke matches at 1.0.
        """
        if best_dislike - best_like >= cfg.match_no_margin:
            return "no"
        if base_score >= cfg.match_yes_min:
            return "yes"
        return "maybe"
