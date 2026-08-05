"""Tests for the deterministic ranking engine."""

import random
from datetime import date, datetime, timezone
from typing import Any

import pytest

from src.config import (
    AppConfig,
    ComfortCurve,
    LocationConfig,
    ScoringConfig,
    ScrapingConfig,
    VenueDiscoveryConfig,
    WeatherConfig,
)
from src.models.event import Event
from src.models.tag import Tag
from src.scoring.ranking import (
    EVERYTHING_ELSE,
    TOP_PICK,
    WORTH_CONSIDERING,
    RankingEngine,
)
from src.scoring.similarity import Reason, SimilarityResult

RUN_DATE = date(2025, 6, 21)
NOW = datetime(2025, 6, 21, 12, 0, tzinfo=timezone.utc)

PLEASANT = {
    "hour": 20,
    "temperature_f": 62.0,
    "dew_point_f": 50.0,
    "wind_speed_mph": 5.0,
    "precipitation_mm": 0.0,
    "condition": "clear",
}

MISERABLE = {
    "hour": 20,
    "temperature_f": 88.0,
    "dew_point_f": 74.0,
    "wind_speed_mph": 18.0,
    "precipitation_mm": 0.0,
    "condition": "thunderstorm",
}


def _weather(hour: dict[str, Any]) -> dict[str, Any]:
    return {
        "sampled_hour": 20,
        "forecast": {"issued_at": NOW.isoformat(), "hour": hour, "day_series": []},
        "observed": None,
    }


def _curve(
    ideal: tuple[float, float],
    zero: tuple[float, float],
    floor: tuple[float, float],
    weight: float = 1.0,
    supersedes: tuple[str, ...] = (),
) -> ComfortCurve:
    return ComfortCurve(
        ideal=ideal, zero=zero, floor=floor, weight=weight, supersedes=supersedes
    )


def _config(**scoring_overrides: Any) -> AppConfig:
    scoring = ScoringConfig(
        top_picks_min=0.5,
        worth_considering_min=0.1,
        match_multiplier_yes=1.5,
        match_multiplier_maybe=1.0,
        match_multiplier_no=0.5,
        min_tags_per_event=5,
    )
    for name, value in scoring_overrides.items():
        setattr(scoring, name, value)

    weather = WeatherConfig(
        max_positive_adjustment=0.15,
        max_negative_adjustment=0.25,
        comfort={
            "temperature_f": _curve((20.0, 65.0), (-15.0, 78.0), (-40.0, 95.0)),
            "dew_point_f": _curve((-99.0, 55.0), (-99.0, 65.0), (-99.0, 75.0)),
            "wind_speed_mph": _curve((0.0, 10.0), (0.0, 20.0), (0.0, 35.0), weight=0.6),
            "precipitation_mm": _curve(
                (0.0, 0.3), (0.0, 2.5), (0.0, 10.0), supersedes=("rain", "snow")
            ),
        },
        condition_penalty={"rain": -0.4, "thunderstorm": -1.0, "clear": 0.0},
    )

    return AppConfig(
        location=LocationConfig(42.52, -70.89, "01970", 10.0, "America/New_York"),
        scraping=ScrapingConfig(),
        venue_discovery=VenueDiscoveryConfig(blocklist_name_match_threshold=0.80),
        scoring=scoring,
        weather=weather,
    )


def _event(
    event_id: str = "evt-1",
    *,
    base_score: float = 0.4,
    match: str = "maybe",
    tag_count: int = 5,
    setting: str = "indoor",
    weather: dict[str, Any] | None = None,
    venue: str | None = "The Jazz Cellar",
    source_type: str = "instagram",
    scored: bool = True,
) -> Event:
    similarity = (
        SimilarityResult(
            tag_score=base_score,
            summary_score=0.0,
            base_score=base_score,
            match=match,
            reasons=[
                Reason(
                    factor="like_similarity",
                    matched_preference="live music",
                    similarity=0.8,
                    contribution=base_score,
                    direction="positive",
                    tag="jazz",
                )
            ],
        )
        if scored
        else None
    )
    return Event(
        event_id=event_id,
        source_event_candidates=[],
        source_type=source_type,
        created_at=NOW,
        updated_at=NOW,
        title="Test Event",
        venue=venue,
        tags=[Tag(text=f"tag{i}") for i in range(tag_count)],
        setting=setting,
        weather=weather,
        similarity=similarity,
    )


def _rank(events: list[Event], config: AppConfig | None = None, blocklist=None):
    engine = RankingEngine(config or _config(), blocklist=blocklist)
    return engine.rank(events, RUN_DATE)


def _scores(events: list[Event], config: AppConfig | None = None) -> dict[str, float]:
    return {r.event_id: r.final_score for r in _rank(events, config)}


# --- ordering ---------------------------------------------------------------


def test_higher_base_score_outranks_lower():
    ranked = _rank([_event("low", base_score=0.1), _event("high", base_score=0.9)])

    assert [r.event_id for r in ranked] == ["high", "low"]


def test_rank_is_one_based_and_contiguous():
    ranked = _rank([_event(f"evt-{i}", base_score=i / 10) for i in range(5)])

    assert [r.rank for r in ranked] == [1, 2, 3, 4, 5]


def test_ties_are_broken_by_event_id():
    """Dedup output order is explicitly unguaranteed, so it must not leak into ranking."""
    ranked = _rank([_event("evt-c"), _event("evt-a"), _event("evt-b")])

    assert [r.event_id for r in ranked] == ["evt-a", "evt-b", "evt-c"]


def test_shuffled_input_produces_identical_output():
    events = [_event(f"evt-{i}", base_score=(i % 3) / 10) for i in range(9)]

    shuffled = list(events)
    random.Random(1).shuffle(shuffled)
    reshuffled = list(events)
    random.Random(2).shuffle(reshuffled)

    assert _rank(shuffled) == _rank(reshuffled)


def test_repeated_runs_are_identical():
    events = [_event("evt-1", base_score=0.4), _event("evt-2", base_score=0.7)]

    assert _rank(events) == _rank(events)


def test_empty_input_returns_nothing():
    assert _rank([]) == []


# --- the match multiplier ---------------------------------------------------


def test_yes_outranks_maybe_outranks_no_from_a_positive_base():
    scores = _scores(
        [
            _event("yes", base_score=0.4, match="yes"),
            _event("maybe", base_score=0.4, match="maybe"),
            _event("no", base_score=0.4, match="no"),
        ]
    )

    assert scores["yes"] > scores["maybe"] > scores["no"]


def test_no_ranks_below_maybe_from_a_negative_base():
    """The direction-aware guard: multiplying a negative base rewards rejection."""
    scores = _scores(
        [
            _event("maybe", base_score=-0.4, match="maybe"),
            _event("no", base_score=-0.4, match="no"),
        ]
    )

    assert scores["no"] < scores["maybe"]


def test_negative_base_is_divided_not_multiplied():
    scores = _scores([_event("no", base_score=-0.4, match="no")])

    assert scores["no"] == pytest.approx(-0.8)


def test_multipliers_are_read_from_config():
    config = _config(match_multiplier_yes=3.0)
    scores = _scores([_event("yes", base_score=0.4, match="yes")], config)

    assert scores["yes"] == pytest.approx(1.2)


# --- tag confidence ---------------------------------------------------------


def test_full_tag_count_scores_at_full_confidence():
    ranked = _rank([_event(tag_count=5, base_score=0.4)])

    assert ranked[0].tag_confidence == pytest.approx(1.0)
    assert ranked[0].final_score == pytest.approx(0.4)


def test_more_tags_than_required_does_not_inflate_the_score():
    ranked = _rank([_event(tag_count=12, base_score=0.4)])

    assert ranked[0].tag_confidence == pytest.approx(1.0)


def test_thin_extraction_scales_the_score_down():
    ranked = _rank([_event(tag_count=2, base_score=0.4)])

    assert ranked[0].tag_confidence == pytest.approx(0.4)
    assert ranked[0].final_score == pytest.approx(0.16)


def test_confidence_pulls_a_negative_score_up_toward_zero():
    """Symmetric, unlike the multiplier: thin evidence means uncertain, not bad."""
    scores = _scores(
        [
            _event("thin", base_score=-0.5, tag_count=1),
            _event("full", base_score=-0.5, tag_count=5),
        ]
    )

    assert scores["thin"] > scores["full"]
    assert scores["thin"] < 0


def test_thin_positive_ranks_below_its_full_confidence_twin():
    scores = _scores(
        [
            _event("thin", base_score=0.5, tag_count=1),
            _event("full", base_score=0.5, tag_count=5),
        ]
    )

    assert scores["thin"] < scores["full"]


def test_no_tags_scores_zero():
    ranked = _rank([_event(tag_count=0, base_score=0.8)])

    assert ranked[0].final_score == pytest.approx(0.0)


def test_min_tags_is_read_from_config():
    ranked = _rank([_event(tag_count=2, base_score=0.4)], _config(min_tags_per_event=2))

    assert ranked[0].tag_confidence == pytest.approx(1.0)


def test_synthetic_activities_are_exempt_from_confidence():
    """Their tags are authored by hand, so a low count is a choice, not a failure."""
    ranked = _rank([_event(tag_count=2, base_score=0.4, source_type="synthetic")])

    assert ranked[0].tag_confidence == pytest.approx(1.0)
    assert ranked[0].final_score == pytest.approx(0.4)


# --- weather ----------------------------------------------------------------


def test_good_weather_lifts_an_outdoor_event_above_its_indoor_twin():
    scores = _scores(
        [
            _event("outdoor", setting="outdoor", weather=_weather(PLEASANT)),
            _event("indoor", setting="indoor", weather=_weather(PLEASANT)),
        ]
    )

    assert scores["outdoor"] > scores["indoor"]


def test_bad_weather_pushes_an_outdoor_event_below_its_indoor_twin():
    scores = _scores(
        [
            _event("outdoor", setting="outdoor", weather=_weather(MISERABLE)),
            _event("indoor", setting="indoor", weather=_weather(MISERABLE)),
        ]
    )

    assert scores["outdoor"] < scores["indoor"]


@pytest.mark.parametrize("setting", ["indoor", "unknown"])
def test_non_outdoor_events_get_no_weather_adjustment(setting):
    ranked = _rank([_event(setting=setting, weather=_weather(PLEASANT))])

    assert ranked[0].weather_adjustment == 0.0
    assert not [r for r in ranked[0].reasons if r.factor == "weather_adjustment"]


def test_missing_weather_is_not_penalised_against_a_peer():
    """Beyond the forecast horizon, an unknown night must not cost the event anything."""
    scores = _scores(
        [
            _event("unknown-weather", setting="outdoor", weather=None),
            _event("indoor", setting="indoor", weather=_weather(PLEASANT)),
        ]
    )

    assert scores["unknown-weather"] == pytest.approx(scores["indoor"])


def test_weather_adjustment_is_recorded_separately_from_the_base():
    ranked = _rank([_event(setting="outdoor", base_score=0.4, weather=_weather(PLEASANT))])

    assert ranked[0].base_score == pytest.approx(0.4)
    assert ranked[0].weather_adjustment > 0
    assert ranked[0].final_score == pytest.approx(0.4 + ranked[0].weather_adjustment)


def test_confidence_does_not_scale_the_weather_adjustment():
    """Weather is measured, not extracted; a thin tag list says nothing about it."""
    thin, full = _rank(
        [
            _event("thin", tag_count=1, setting="outdoor", weather=_weather(PLEASANT)),
            _event("full", tag_count=5, setting="outdoor", weather=_weather(PLEASANT)),
        ]
    )

    assert thin.weather_adjustment == pytest.approx(full.weather_adjustment)


# --- tiers ------------------------------------------------------------------


def test_tiers_are_assigned_from_the_configured_thresholds():
    ranked = _rank(
        [
            _event("top", base_score=0.9),
            _event("middle", base_score=0.3),
            _event("bottom", base_score=0.05),
        ]
    )
    tiers = {r.event_id: r.tier for r in ranked}

    assert tiers == {
        "top": TOP_PICK,
        "middle": WORTH_CONSIDERING,
        "bottom": EVERYTHING_ELSE,
    }


def test_threshold_boundaries_belong_to_the_higher_tier():
    ranked = _rank([_event("at-top", base_score=0.5), _event("at-worth", base_score=0.1)])
    tiers = {r.event_id: r.tier for r in ranked}

    assert tiers == {"at-top": TOP_PICK, "at-worth": WORTH_CONSIDERING}


def test_tier_thresholds_are_read_from_config():
    ranked = _rank([_event(base_score=0.3)], _config(top_picks_min=0.2))

    assert ranked[0].tier == TOP_PICK


def test_a_below_threshold_event_is_present_not_dropped():
    ranked = _rank([_event("bad", base_score=-2.0)])

    assert [r.event_id for r in ranked] == ["bad"]
    assert ranked[0].tier == EVERYTHING_ELSE


# --- blocklist and completeness ---------------------------------------------


def test_blocklisted_venue_never_appears():
    ranked = _rank(
        [_event("blocked", venue="The Sports Bar"), _event("kept", venue="Jazz Cellar")],
        blocklist=["The Sports Bar"],
    )

    assert [r.event_id for r in ranked] == ["kept"]


def test_blocklist_matches_a_near_miss_venue_name():
    ranked = _rank([_event("blocked", venue="the sports bar")], blocklist=["The Sports Bar"])

    assert ranked == []


def test_event_without_a_venue_is_never_blocked():
    ranked = _rank([_event("no-venue", venue=None)], blocklist=["The Sports Bar"])

    assert [r.event_id for r in ranked] == ["no-venue"]


def test_nothing_but_the_blocklist_is_ever_dropped():
    events = [
        _event("negative", base_score=-3.0),
        _event("zero", base_score=0.0),
        _event("positive", base_score=2.0),
        _event("blocked", venue="The Sports Bar"),
    ]

    ranked = _rank(events, blocklist=["The Sports Bar"])

    assert len(ranked) == 3


def test_ranks_stay_contiguous_after_a_blocklist_drop():
    ranked = _rank(
        [
            _event("a", base_score=0.9),
            _event("blocked", base_score=0.8, venue="The Sports Bar"),
            _event("b", base_score=0.7),
        ],
        blocklist=["The Sports Bar"],
    )

    assert [r.rank for r in ranked] == [1, 2]


# --- the record itself ------------------------------------------------------


def test_run_date_comes_from_the_caller():
    """No clock in the engine, or the determinism guarantee is untestable."""
    assert _rank([_event()])[0].run_date == RUN_DATE


def test_match_label_is_carried_through():
    assert _rank([_event(match="yes")])[0].match == "yes"


def test_similarity_reasons_are_carried_through():
    factors = [r.factor for r in _rank([_event()])[0].reasons]

    assert "like_similarity" in factors


def test_the_multiplier_is_explained():
    """A score that moves without a reason is unauditable."""
    reasons = _rank([_event(match="yes", base_score=0.4)])[0].reasons
    classification = [r for r in reasons if r.factor == "match_classification"]

    assert len(classification) == 1
    assert classification[0].contribution == pytest.approx(0.2)


def test_low_confidence_is_explained():
    reasons = _rank([_event(tag_count=2, base_score=0.4)])[0].reasons

    assert [r for r in reasons if r.factor == "low_tag_confidence"]


def test_full_confidence_emits_no_confidence_reason():
    reasons = _rank([_event(tag_count=5)])[0].reasons

    assert not [r for r in reasons if r.factor == "low_tag_confidence"]


def test_weather_is_explained_when_it_applies():
    reasons = _rank([_event(setting="outdoor", weather=_weather(PLEASANT))])[0].reasons

    assert [r for r in reasons if r.factor == "weather_adjustment"]


def test_recommendation_id_is_derived_from_the_run_and_event():
    first = _rank([_event("evt-1")])[0]
    second = _rank([_event("evt-1")])[0]

    assert first.recommendation_id == second.recommendation_id


# --- degraded input ---------------------------------------------------------


def test_unscored_event_ranks_at_zero_rather_than_raising():
    """One unscorable event costs one recommendation, not the batch."""
    ranked = _rank([_event("unscored", scored=False), _event("scored", base_score=0.5)])

    assert [r.event_id for r in ranked] == ["scored", "unscored"]
    assert ranked[1].base_score == pytest.approx(0.0)


def test_unscored_event_is_labelled_maybe():
    ranked = _rank([_event("unscored", scored=False)])

    assert ranked[0].match == "maybe"
