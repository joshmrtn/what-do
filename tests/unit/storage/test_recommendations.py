"""Unit tests for recommendation persistence."""

from datetime import date
from pathlib import Path

import pytest

from src.models.recommendation import Recommendation, make_recommendation_id
from src.scoring.similarity import Reason
from src.storage.db import init_db
from src.storage.recommendations import load_recommendations, save_recommendations

RUN_DATE = date(2025, 6, 21)
NEXT_DAY = date(2025, 6, 22)


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    p = tmp_path / "test.db"
    init_db(p)
    return p


def _reason(factor: str = "like_similarity", contribution: float = 0.8) -> Reason:
    return Reason(
        factor=factor,
        matched_preference="karaoke night",
        similarity=0.87,
        contribution=contribution,
        direction="positive",
        tag="karaoke",
    )


def _make(event_id: str = "evt-1", run_date: date = RUN_DATE, rank: int = 1, **overrides):
    fields = {
        "recommendation_id": make_recommendation_id(run_date, event_id),
        "event_id": event_id,
        "run_date": run_date,
        "base_score": 0.42,
        "weather_adjustment": 0.05,
        "tag_confidence": 1.0,
        "final_score": 0.68,
        "match": "yes",
        "tier": "top_pick",
        "rank": rank,
        "reasons": [_reason()],
    }
    fields.update(overrides)
    return Recommendation(**fields)


def test_round_trip_preserves_every_field(db_path):
    original = _make()
    save_recommendations([original], db_path)

    assert load_recommendations(db_path) == [original]


def test_score_components_survive_the_round_trip(db_path):
    """base_score and weather_adjustment explain final_score; a lost term is unexplainable."""
    save_recommendations(
        [_make(base_score=-0.3, weather_adjustment=-0.25, final_score=-0.85, tag_confidence=0.4)],
        db_path,
    )
    loaded = load_recommendations(db_path)[0]

    assert loaded.base_score == pytest.approx(-0.3)
    assert loaded.weather_adjustment == pytest.approx(-0.25)
    assert loaded.tag_confidence == pytest.approx(0.4)
    assert loaded.final_score == pytest.approx(-0.85)


def test_reasons_load_as_structured_objects(db_path):
    save_recommendations([_make(reasons=[_reason(), _reason(factor="weather_adjustment")])], db_path)
    reasons = load_recommendations(db_path)[0].reasons

    assert [r.factor for r in reasons] == ["like_similarity", "weather_adjustment"]
    assert all(isinstance(r, Reason) for r in reasons)


def test_rerunning_the_same_date_replaces_that_runs_rows(db_path):
    save_recommendations([_make(event_id="evt-1"), _make(event_id="evt-2", rank=2)], db_path)
    save_recommendations([_make(event_id="evt-1", final_score=0.9)], db_path)

    loaded = load_recommendations(db_path)
    assert len(loaded) == 1
    assert loaded[0].final_score == pytest.approx(0.9)


def test_replacement_is_scoped_to_the_run_date(db_path):
    """Re-ranking tonight must not erase last night's decisions."""
    save_recommendations([_make(event_id="evt-1", run_date=RUN_DATE)], db_path)
    save_recommendations([_make(event_id="evt-2", run_date=NEXT_DAY)], db_path)

    assert {r.run_date for r in load_recommendations(db_path)} == {RUN_DATE, NEXT_DAY}


def test_load_filters_by_run_date(db_path):
    save_recommendations([_make(event_id="evt-1", run_date=RUN_DATE)], db_path)
    save_recommendations([_make(event_id="evt-2", run_date=NEXT_DAY)], db_path)

    loaded = load_recommendations(db_path, run_date=RUN_DATE)
    assert [r.event_id for r in loaded] == ["evt-1"]


def test_rows_load_in_rank_order(db_path):
    """The batch's ordering is the product; reading must not reshuffle it."""
    save_recommendations(
        [
            _make(event_id="evt-c", rank=3),
            _make(event_id="evt-a", rank=1),
            _make(event_id="evt-b", rank=2),
        ],
        db_path,
    )

    assert [r.rank for r in load_recommendations(db_path)] == [1, 2, 3]


def test_saving_nothing_is_a_no_op(db_path):
    save_recommendations([], db_path)
    assert load_recommendations(db_path) == []


def test_saving_nothing_does_not_clear_an_existing_run(db_path):
    """An empty batch is not an instruction to delete what is already there."""
    save_recommendations([_make()], db_path)
    save_recommendations([], db_path)

    assert len(load_recommendations(db_path)) == 1


def test_run_date_round_trips_as_a_date(db_path):
    save_recommendations([_make()], db_path)
    assert load_recommendations(db_path)[0].run_date == RUN_DATE
