"""Unit tests for the similarity engine and match classifier."""

from __future__ import annotations

from datetime import datetime, timezone
import math

import pytest

from src.config import ScoringConfig
from src.models.event import Event
from src.models.tag import Tag
from src.scoring.preferences import PreferenceSet, UserPreference
from src.scoring.similarity import SimilarityEngine
from src.utils.vectors import encode_vector

_NOW = datetime(2026, 6, 15, tzinfo=timezone.utc)


def _at(similarity: float) -> list[float]:
    """A unit vector whose cosine with _at(1.0) is exactly `similarity`."""
    return [similarity, math.sqrt(max(0.0, 1.0 - similarity**2))]


def _like(text: str, similarity: float, domain: str = "general") -> UserPreference:
    return UserPreference("like", domain, text, _at(similarity))


def _dislike(text: str, similarity: float, domain: str = "general") -> UserPreference:
    return UserPreference("dislike", domain, text, _at(similarity))


#: Two orthogonal tag directions, so a tag can sit near the likes or near the
#: dislikes. LIKED is the reference direction all _at() preferences measure from.
LIKED = [1.0, 0.0]
DISLIKED = [0.0, 1.0]


def _event(
    tags, *, source_type="apify", summary=None, summary_similarity=None, directions=None
) -> Event:
    """Event whose tags sit along LIKED unless `directions` says otherwise."""
    event = Event(
        event_id="e1",
        source_event_candidates=[],
        source_type=source_type,
        created_at=_NOW,
        updated_at=_NOW,
        tags=tags,
        summary=summary,
    )
    vectors = directions if directions is not None else [LIKED] * len(tags)
    event.tag_embeddings = [encode_vector(v) for v in vectors]
    if summary_similarity is not None:
        event.summary_embedding = encode_vector(_at(summary_similarity))
    return event


def _cfg(**kwargs) -> ScoringConfig:
    return ScoringConfig(**kwargs)


def _score(event, likes=(), dislikes=(), cfg=None):
    prefs = PreferenceSet(likes=list(likes), dislikes=list(dislikes))
    return SimilarityEngine().score(event, prefs, cfg or _cfg())


# ---------------------------------------------------------------------------
# Specificity wins — direction follows the closer match
# ---------------------------------------------------------------------------


def test_tag_closer_to_a_like_contributes_positively():
    result = _score(
        _event([Tag("karaoke")]),
        likes=[_like("karaoke", 0.95)],
        dislikes=[_dislike("bars", 0.70)],
    )

    assert result.tag_score > 0


def test_tag_closer_to_a_dislike_contributes_negatively():
    result = _score(
        _event([Tag("bar")]),
        likes=[_like("karaoke", 0.70)],
        dislikes=[_dislike("bars", 0.95)],
    )

    assert result.tag_score < 0


def test_strongest_preference_on_each_side_is_the_one_used():
    """max, not sum — so the score does not depend on how many synonyms exist."""
    result = _score(
        _event([Tag("karaoke")]),
        likes=[_like("karaoke", 0.95), _like("singing", 0.80), _like("music", 0.75)],
        dislikes=[_dislike("bars", 0.50)],
    )

    assert result.reasons[0].matched_preference == "karaoke"
    assert result.reasons[0].similarity == pytest.approx(0.95)


# ---------------------------------------------------------------------------
# The logistic gate — noise must not vote
# ---------------------------------------------------------------------------


def test_noise_level_similarities_contribute_almost_nothing():
    """Unrelated pairs sit near 0.42; a coin-flip between them must not score."""
    result = _score(
        _event([Tag("sushi")]),
        likes=[_like("karaoke", 0.407)],
        dislikes=[_dislike("dancing", 0.433)],
    )

    assert abs(result.tag_score) < 0.02


def test_gate_midpoint_is_configurable():
    event = _event([Tag("karaoke")])
    prefs = dict(likes=[_like("karaoke", 0.65)], dislikes=[])

    low = _score(event, **prefs, cfg=_cfg(gate_midpoint=0.50))
    high = _score(event, **prefs, cfg=_cfg(gate_midpoint=0.90))

    assert low.tag_score > high.tag_score


def test_strong_match_passes_the_gate_nearly_intact():
    result = _score(_event([Tag("karaoke")]), likes=[_like("karaoke", 0.95)])

    assert result.tag_score == pytest.approx(0.95, abs=0.01)


# ---------------------------------------------------------------------------
# Centrality weights multiply the contribution
# ---------------------------------------------------------------------------


def test_weight_scales_the_contribution():
    full = _score(_event([Tag("bar", 1.0)]), dislikes=[_dislike("bars", 0.95)])
    incidental = _score(_event([Tag("bar", 0.2)]), dislikes=[_dislike("bars", 0.95)])

    assert abs(incidental.tag_score) < abs(full.tag_score)
    assert incidental.tag_score == pytest.approx(full.tag_score * 0.2, abs=0.01)


def test_incidental_dislike_does_not_outweigh_a_central_like():
    """The karaoke-at-a-bar case: 'bar' is real but not what the event is."""
    result = _score(
        _event([Tag("karaoke", 1.0), Tag("bar", 0.2)]),
        likes=[_like("karaoke", 0.95)],
        dislikes=[_dislike("bars", 0.93)],
    )

    assert result.tag_score > 0


def test_zero_weight_tag_contributes_nothing():
    result = _score(_event([Tag("bar", 0.0)]), dislikes=[_dislike("bars", 0.95)])

    assert result.tag_score == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def test_balanced_mean_lets_one_strong_negative_outweigh_several_weak_positives():
    result = _score(
        _event([Tag("a"), Tag("b"), Tag("c"), Tag("bar")]),
        likes=[_like("mild", 0.70)],
        dislikes=[_dislike("bars", 0.99)],
        cfg=_cfg(aggregator="balanced_mean"),
    )

    assert result.tag_score < 0


def _mixed_event() -> Event:
    """One strongly-liked tag against three strongly-disliked ones."""
    return _event(
        [Tag("karaoke"), Tag("bar"), Tag("drinks"), Tag("nightlife")],
        directions=[LIKED, DISLIKED, DISLIKED, DISLIKED],
    )


def _mixed_prefs() -> dict:
    return dict(
        likes=[UserPreference("like", "general", "karaoke", [0.95, 0.3122])],
        dislikes=[UserPreference("dislike", "general", "bars", [0.3122, 0.95])],
    )


def test_specificity_sum_dilutes_by_negative_tag_count():
    """The originally specified aggregator: negatives accumulate and swamp."""
    result = _score(_mixed_event(), **_mixed_prefs(), cfg=_cfg(aggregator="specificity_sum"))

    assert result.tag_score < -0.4


def test_balanced_mean_compares_average_strength_not_counts():
    """Same event, same evidence — three incidental negatives no longer swamp one positive."""
    result = _score(_mixed_event(), **_mixed_prefs(), cfg=_cfg(aggregator="balanced_mean"))

    assert result.tag_score == pytest.approx(0.0, abs=0.01)


def test_aggregator_selected_from_config():
    balanced = _score(_mixed_event(), **_mixed_prefs(), cfg=_cfg(aggregator="balanced_mean"))
    summed = _score(_mixed_event(), **_mixed_prefs(), cfg=_cfg(aggregator="specificity_sum"))

    assert balanced.tag_score > summed.tag_score


def test_all_positive_event_scores_positive():
    result = _score(
        _event([Tag("karaoke"), Tag("singing")]), likes=[_like("karaoke", 0.95)]
    )

    assert result.tag_score > 0


# ---------------------------------------------------------------------------
# Summary term
# ---------------------------------------------------------------------------


def test_summary_contributes_at_the_configured_weight():
    event = _event([Tag("karaoke")], summary="A karaoke night.", summary_similarity=0.95)

    weighted = _score(event, likes=[_like("karaoke", 0.95)], cfg=_cfg(summary_weight=0.3))
    ignored = _score(event, likes=[_like("karaoke", 0.95)], cfg=_cfg(summary_weight=0.0))

    assert weighted.base_score > ignored.base_score
    assert ignored.base_score == pytest.approx(ignored.tag_score)


def test_base_score_is_tag_score_plus_weighted_summary():
    event = _event([Tag("karaoke")], summary="A karaoke night.", summary_similarity=0.90)
    result = _score(event, likes=[_like("karaoke", 0.95)], cfg=_cfg(summary_weight=0.3))

    assert result.base_score == pytest.approx(
        result.tag_score + 0.3 * result.summary_score
    )


def test_event_without_summary_embedding_scores_on_tags_alone():
    result = _score(_event([Tag("karaoke")]), likes=[_like("karaoke", 0.95)])

    assert result.summary_score == 0.0
    assert result.base_score == pytest.approx(result.tag_score)


# ---------------------------------------------------------------------------
# Domain scoping
# ---------------------------------------------------------------------------


def test_domain_preference_ignored_for_unmapped_source_type():
    result = _score(
        _event([Tag("horror")], source_type="apify"),
        likes=[_like("horror films", 0.95, domain="movies")],
    )

    assert result.tag_score == pytest.approx(0.0)
    assert result.reasons == []


def test_domain_preference_applied_to_mapped_source_type():
    result = _score(
        _event([Tag("horror")], source_type="cinema_veezi"),
        likes=[_like("horror films", 0.95, domain="movies")],
        cfg=_cfg(domain_map={"cinema_veezi": "movies"}),
    )

    assert result.tag_score > 0


def test_general_preferences_apply_to_every_source_type():
    for source_type in ("apify", "cinema_veezi", "synthetic"):
        result = _score(
            _event([Tag("karaoke")], source_type=source_type),
            likes=[_like("karaoke", 0.95)],
            cfg=_cfg(domain_map={"cinema_veezi": "movies"}),
        )
        assert result.tag_score > 0, source_type


# ---------------------------------------------------------------------------
# Reason objects
# ---------------------------------------------------------------------------


def test_reason_carries_the_full_schema():
    result = _score(_event([Tag("karaoke", 0.8)]), likes=[_like("karaoke night", 0.95)])
    reason = result.reasons[0]

    assert reason.factor == "like_similarity"
    assert reason.tag == "karaoke"
    assert reason.matched_preference == "karaoke night"
    assert reason.similarity == pytest.approx(0.95)
    assert reason.contribution == pytest.approx(0.95 * 0.8, abs=0.01)
    assert reason.direction == "positive"


def test_negative_reason_uses_dislike_factor_and_direction():
    result = _score(_event([Tag("bar")]), dislikes=[_dislike("bars", 0.95)])
    reason = result.reasons[0]

    assert reason.factor == "dislike_similarity"
    assert reason.direction == "negative"
    assert reason.contribution < 0


def test_one_reason_per_scored_tag():
    result = _score(
        _event([Tag("karaoke"), Tag("bar")]),
        likes=[_like("karaoke", 0.95)],
        dislikes=[_dislike("bars", 0.90)],
    )

    assert [r.tag for r in result.reasons] == ["karaoke", "bar"]


def test_summary_reason_has_no_tag():
    event = _event([Tag("karaoke")], summary="A karaoke night.", summary_similarity=0.92)
    result = _score(event, likes=[_like("karaoke", 0.95)])

    summary_reasons = [r for r in result.reasons if r.tag is None]
    assert len(summary_reasons) == 1
    assert summary_reasons[0].matched_preference == "karaoke"


# ---------------------------------------------------------------------------
# Match classification — relative margin, never an absolute dislike cutoff
# ---------------------------------------------------------------------------


def test_strong_dislike_beating_the_best_like_classifies_no():
    result = _score(
        _event([Tag("sports bar")]),
        likes=[_like("karaoke", 0.43)],
        dislikes=[_dislike("bars", 0.85)],
    )

    assert result.match == "no"


def test_strong_dislike_that_loses_to_a_stronger_like_is_not_no():
    """Koto: 'bar' scores 0.932 against 'bars', but karaoke matches at 1.0."""
    result = _score(
        _event([Tag("karaoke", 1.0), Tag("bar", 0.6)]),
        likes=[_like("karaoke", 1.0)],
        dislikes=[_dislike("bars", 0.932)],
    )

    assert result.match != "no"


def test_high_scoring_event_classifies_yes():
    result = _score(
        _event([Tag("karaoke"), Tag("punk")]),
        likes=[_like("karaoke", 0.95)],
        cfg=_cfg(match_yes_min=0.30),
    )

    assert result.match == "yes"


def test_middling_event_classifies_maybe():
    result = _score(
        _event([Tag("karaoke")]),
        likes=[_like("karaoke", 0.70)],
        cfg=_cfg(match_yes_min=0.90),
    )

    assert result.match == "maybe"


def test_match_thresholds_read_from_config():
    event = _event([Tag("karaoke")])
    prefs = dict(likes=[_like("karaoke", 0.80)])

    assert _score(event, **prefs, cfg=_cfg(match_yes_min=0.10)).match == "yes"
    assert _score(event, **prefs, cfg=_cfg(match_yes_min=0.99)).match == "maybe"


def test_no_margin_read_from_config():
    event = _event([Tag("bar")])
    prefs = dict(likes=[_like("karaoke", 0.70)], dislikes=[_dislike("bars", 0.80)])

    assert _score(event, **prefs, cfg=_cfg(match_no_margin=0.05)).match == "no"
    assert _score(event, **prefs, cfg=_cfg(match_no_margin=0.50)).match != "no"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_no_preferences_at_all_scores_zero():
    result = _score(_event([Tag("karaoke")]))

    assert result.base_score == 0.0
    assert result.reasons == []
    assert result.match == "maybe"


def test_only_dislikes_configured_still_scores():
    result = _score(_event([Tag("bar")]), dislikes=[_dislike("bars", 0.95)])

    assert result.tag_score < 0


def test_event_without_tag_embeddings_scores_zero_on_tags():
    event = _event([])

    result = _score(event, likes=[_like("karaoke", 0.95)])

    assert result.tag_score == 0.0


def test_tag_count_mismatch_with_embeddings_is_tolerated():
    """Defensive: a partially embedded event must not raise mid-batch."""
    event = _event([Tag("karaoke"), Tag("bar")])
    event.tag_embeddings = event.tag_embeddings[:1]

    result = _score(event, likes=[_like("karaoke", 0.95)])

    assert len(result.reasons) == 1


def test_scoring_is_deterministic():
    event = _event([Tag("karaoke"), Tag("bar")])
    prefs = dict(likes=[_like("karaoke", 0.95)], dislikes=[_dislike("bars", 0.90)])

    first, second = _score(event, **prefs), _score(event, **prefs)

    assert first.base_score == second.base_score
    assert [r.contribution for r in first.reasons] == [r.contribution for r in second.reasons]


def test_result_attached_to_event_by_the_engine_caller():
    """The engine is pure; the stage attaches. Verify the field exists and types."""
    event = _event([Tag("karaoke")])
    result = _score(event, likes=[_like("karaoke", 0.95)])

    event.similarity = result
    assert event.similarity.base_score == result.base_score
