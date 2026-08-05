"""Unit tests for the Recommendation model."""

from datetime import date

import pytest

from src.models.recommendation import (
    Recommendation,
    make_recommendation_id,
    reasons_from_json,
    reasons_to_json,
)
from src.scoring.similarity import Reason

RUN_DATE = date(2025, 6, 21)


def _reason(factor: str = "like_similarity", contribution: float = 0.8) -> Reason:
    return Reason(
        factor=factor,
        matched_preference="karaoke night",
        similarity=0.87,
        contribution=contribution,
        direction="positive",
        tag="karaoke",
    )


def _make(**overrides) -> Recommendation:
    fields = {
        "recommendation_id": make_recommendation_id(RUN_DATE, "evt-1"),
        "event_id": "evt-1",
        "run_date": RUN_DATE,
        "base_score": 0.42,
        "weather_adjustment": 0.05,
        "tag_confidence": 1.0,
        "final_score": 0.68,
        "match": "yes",
        "tier": "top_pick",
        "rank": 1,
        "reasons": [_reason()],
    }
    fields.update(overrides)
    return Recommendation(**fields)


def test_recommendation_is_frozen():
    """A ranked decision is a record of a run, not a mutable working value."""
    with pytest.raises(Exception):
        _make().final_score = 0.9


def test_id_is_stable_for_the_same_run_and_event():
    """Two runs of the same batch must be identical, which a random id prevents."""
    first = make_recommendation_id(RUN_DATE, "evt-1")
    second = make_recommendation_id(RUN_DATE, "evt-1")
    assert first == second


def test_id_differs_by_event():
    assert make_recommendation_id(RUN_DATE, "evt-1") != make_recommendation_id(RUN_DATE, "evt-2")


def test_id_differs_by_run_date():
    """The same event ranked on two nights is two decisions, not one."""
    assert make_recommendation_id(RUN_DATE, "evt-1") != make_recommendation_id(
        date(2025, 6, 22), "evt-1"
    )


def test_reasons_round_trip_through_json():
    original = [_reason(), _reason(factor="dislike_similarity", contribution=-0.3)]
    restored = reasons_from_json(reasons_to_json(original))

    assert restored == original


def test_reasons_from_json_yields_reason_objects_not_dicts():
    restored = reasons_from_json(reasons_to_json([_reason()]))

    assert isinstance(restored[0], Reason)
    assert restored[0].factor == "like_similarity"


def test_reason_with_no_tag_round_trips():
    """The summary term carries no tag; None must survive as None."""
    summary_reason = Reason(
        factor="like_similarity",
        matched_preference="live music",
        similarity=0.7,
        contribution=0.7,
        direction="positive",
        tag=None,
    )
    assert reasons_from_json(reasons_to_json([summary_reason]))[0].tag is None


def test_empty_reasons_round_trip():
    assert reasons_from_json(reasons_to_json([])) == []
