"""The read-time rescore, end to end against real SQLite.

Everything real except the forecast, which is the one external boundary on this
path. In particular the pipeline stages are the real ones, so what these tests
constrain is what a `what-do` invocation actually does.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from src.composition.storage import build_view_storage
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
from src.models.event_score import EventScore
from src.models.ranking import Ranking
from src.models.tag import Tag
from src.presentation.rescore import rescore_if_stale
from src.scoring.embedding_stage import embedding_input_hash
from src.scoring.similarity import Reason, SimilarityResult
from src.utils.vectors import encode_vector
from src.storage.events import save_events
from src.storage.queries import load_ranked_events
from src.storage.sqlite.connection import init_db
from src.utils.logging import get_logger

TZ = timezone(timedelta(hours=-4))
RUN_DATE = date(2026, 8, 17)
NOW = datetime(2026, 8, 17, 18, 0, tzinfo=TZ)
STALE_ISSUE = NOW - timedelta(hours=14)
TTL = timedelta(minutes=60)


def _hour(temperature: float, condition: str) -> dict[str, Any]:
    return {
        "hour": 20,
        "temperature_f": temperature,
        "dew_point_f": 50.0,
        "wind_speed_mph": 5.0,
        "precipitation_mm": 0.0,
        "condition": condition,
    }


class _FixedWeather:
    """Returns one day of hourly weather. A boundary, so a fake belongs here."""

    def __init__(self, hour: dict[str, Any]) -> None:
        self._hour = hour
        self.calls = 0

    def fetch(self, day: date, lat: float, lng: float) -> dict[str, Any]:
        self.calls += 1
        return {"date": day.isoformat(), "hours": [self._hour]}


class _RefusingWeather:
    """Stands in for the network being down."""

    def fetch(self, day: date, lat: float, lng: float) -> dict[str, Any] | None:
        raise OSError("no route to host")


def _curve(ideal, zero, floor, weight=1.0, supersedes=()):
    return ComfortCurve(
        ideal=ideal, zero=zero, floor=floor, weight=weight, supersedes=supersedes
    )


def _config() -> AppConfig:
    return AppConfig(
        location=LocationConfig(42.52, -70.89, "01970", 10.0, "America/New_York"),
        scraping=ScrapingConfig(),
        venue_discovery=VenueDiscoveryConfig(blocklist_name_match_threshold=0.80),
        scoring=ScoringConfig(
            match_multiplier_yes=1.5,
            match_multiplier_maybe=1.0,
            match_multiplier_no=0.5,
            min_tags_per_event=1,
        ),
        weather=WeatherConfig(
            max_positive_adjustment=0.15,
            max_negative_adjustment=0.25,
            comfort={
                "temperature_f": _curve((20.0, 65.0), (-15.0, 78.0), (-40.0, 95.0)),
            },
            condition_penalty={"rain": -0.4, "thunderstorm": -1.0, "clear": 0.0},
        ),
    )


def _event(event_id: str, title: str, setting: str, base: float) -> Event:
    """An event as a completed batch leaves it — vectors attached.

    The vectors are the point. Without them `EmbeddingStage` reaches for the
    refusing provider, and the first cut of these tests passed while every
    semantic score had been zeroed: the right order for the wrong reason.
    """
    event = Event(
        event_id=event_id,
        source_event_candidates=[f"cand-{event_id}"],
        source_type="instagram",
        created_at=NOW,
        updated_at=NOW,
        title=title,
        venue="The Pier",
        start_time=datetime(2026, 8, 17, 20, 0, tzinfo=TZ),
        setting=setting,
        tags=[Tag(text="music")],
        summary=f"{title} at The Pier",
        weather={
            "sampled_hour": 20,
            "forecast": {
                "issued_at": STALE_ISSUE.isoformat(),
                "hour": _hour(66.0, "clear"),
                "day_series": [],
            },
            "observed": None,
        },
        similarity=SimilarityResult(
            tag_score=base,
            summary_score=0.0,
            base_score=base,
            match="yes",
            reasons=[
                Reason(
                    factor="like_similarity",
                    matched_preference="live music",
                    similarity=0.8,
                    contribution=base,
                    direction="positive",
                    tag="music",
                )
            ],
        ),
    )
    # A vector per tag, plus one for the summary, exactly as storage holds them.
    # `embedding_input_hash` is taken last so the stage recognises the event as
    # already embedded and returns at its early exit, which is what makes the
    # read path free of model calls.
    event.attach_tag_embeddings([encode_vector([0.1, 0.2, 0.3])])
    event.summary_embedding = encode_vector([0.4, 0.5, 0.6])
    event.embedding_input_hash = embedding_input_hash(event)
    return event


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    """Two events a hair apart, one outdoors, ranked by a batch."""
    path = tmp_path / "event_hub.db"
    init_db(path)

    outdoor = _event("outdoor", "Harbour Fireworks", "outdoor", 0.50)
    indoor = _event("indoor", "Cellar Jazz", "indoor", 0.52)
    save_events([outdoor, indoor], path)

    scores = [
        EventScore(
            event_id=event.event_id,
            run_date=RUN_DATE,
            base_score=base,
            tag_confidence=1.0,
            match="yes",
            reasons=[],
        )
        for event, base in ((outdoor, 0.50), (indoor, 0.52))
    ]
    rankings = [
        Ranking(
            event_id="indoor",
            run_date=RUN_DATE,
            weather_adjustment=0.0,
            final_score=0.52,
            rank=1,
        ),
        Ranking(
            event_id="outdoor",
            run_date=RUN_DATE,
            weather_adjustment=0.0,
            final_score=0.50,
            rank=2,
        ),
    ]
    storage = build_view_storage(path, "nomic-embed-text")
    storage.scores.save(scores)
    storage.rankings.save(rankings)
    return path


def _rescore(
    db_path: Path,
    weather: Any,
    *,
    now: datetime = NOW,
    tonight: date = RUN_DATE,
) -> Any:
    storage = build_view_storage(db_path, "nomic-embed-text")
    pairs = load_ranked_events(storage.events, storage.scores, storage.rankings)
    return rescore_if_stale(
        pairs=pairs,
        tonight=tonight,
        now=now,
        ttl=TTL,
        config=_config(),
        db_path=db_path,
        storage=storage,
        logger=get_logger("rescore_test"),
        get_now=lambda: now,
        weather_provider=weather,
        likes_path=db_path.parent / "likes.txt",
        dislikes_path=db_path.parent / "dislikes.txt",
        blocklist_path=db_path.parent / "blocklist.json",
    )


def test_a_pleasant_forecast_lifts_the_outdoor_event_above_the_indoor_one(db_path):
    """The order is the product, and this is the whole point of the feature.

    The batch ranked the indoor event first on a hair's difference and a weather
    adjustment of zero. Given a real forecast the outdoor event earns a positive
    adjustment and overtakes it.
    """
    refreshed = _rescore(db_path, _FixedWeather(_hour(66.0, "clear")))

    assert refreshed is not None
    assert [pair.event.event_id for pair in refreshed] == ["outdoor", "indoor"]


def test_the_new_adjustment_is_stored_not_just_displayed(db_path):
    """`--explain` must quote the scoring the reader just saw."""
    _rescore(db_path, _FixedWeather(_hour(66.0, "clear")))

    storage = build_view_storage(db_path, "nomic-embed-text")
    stored = {r.event_id: r for r in storage.rankings.for_run(RUN_DATE)}

    assert stored["outdoor"].weather_adjustment > 0.0
    assert stored["indoor"].weather_adjustment == 0.0


def test_the_reasons_describe_the_new_readings_not_the_old_ones(db_path):
    """The half the stub missed: the explanation lives on `EventScore.reasons`.

    Rewriting the number and leaving these behind is the two-numbers-two-
    questions failure the rank-display work removed.
    """
    _rescore(db_path, _FixedWeather(_hour(66.0, "clear")))

    storage = build_view_storage(db_path, "nomic-embed-text")
    outdoor = next(
        score
        for score in storage.scores.for_run(RUN_DATE)
        if score.event_id == "outdoor"
    )

    assert any(reason.factor == "weather_adjustment" for reason in outdoor.reasons)


def test_a_foul_forecast_pushes_the_outdoor_event_down(db_path):
    """The adjustment is signed, so the rescore can demote as well as promote."""
    refreshed = _rescore(db_path, _FixedWeather(_hour(94.0, "thunderstorm")))

    assert refreshed is not None
    assert [pair.event.event_id for pair in refreshed] == ["indoor", "outdoor"]
    storage = build_view_storage(db_path, "nomic-embed-text")
    stored = {r.event_id: r for r in storage.rankings.for_run(RUN_DATE)}
    assert stored["outdoor"].weather_adjustment < 0.0


def test_the_rescore_is_recorded(db_path):
    """Otherwise a run date holds numbers its own history does not describe."""
    _rescore(db_path, _FixedWeather(_hour(66.0, "clear")))

    storage = build_view_storage(db_path, "nomic-embed-text")
    recorded = storage.rescores.latest_for(RUN_DATE)

    assert recorded is not None
    assert recorded.rescored_at == NOW
    assert recorded.events_rescored == 2


def test_a_second_invocation_opens_no_connection(db_path):
    """The forecast is cached, so the follow-up is instant.

    Asserted on the provider's call count rather than on elapsed time, which
    would be a flaky way to say the same thing.
    """
    weather = _FixedWeather(_hour(66.0, "clear"))
    _rescore(db_path, weather)
    after_first = weather.calls

    _rescore(db_path, weather, now=NOW + timedelta(minutes=5))

    assert after_first == 1
    assert weather.calls == 1, "the second rescore refetched the forecast"


def test_a_network_failure_leaves_the_stored_ranking_alone(db_path):
    """A rescore may never cost the listing."""
    assert _rescore(db_path, _RefusingWeather()) is None

    storage = build_view_storage(db_path, "nomic-embed-text")
    stored = storage.rankings.for_run(RUN_DATE)

    assert [r.event_id for r in sorted(stored, key=lambda r: r.rank)] == [
        "indoor",
        "outdoor",
    ]


def test_a_ranking_from_another_night_is_not_touched(db_path):
    """`scope_filter` derives its floor from the run date, so mixing is wrong."""
    weather = _FixedWeather(_hour(66.0, "clear"))

    assert _rescore(db_path, weather, tonight=RUN_DATE + timedelta(days=1)) is None
    assert weather.calls == 0, "it fetched before deciding it had nothing to do"


def test_an_event_needing_a_vector_abandons_the_whole_rescore(db_path):
    """The seal is the refusing provider *plus* this check.

    `EmbeddingStage` degrades per event by design — it marks the event and
    carries on, so one bad event never costs a batch the rest. On the read path
    that is exactly wrong: the event scores zero on its semantic half and the
    rescore writes that zero over a good stored score.
    """
    storage = build_view_storage(db_path, "nomic-embed-text")
    stored = {event.event_id: event for event in storage.events.load_all()}
    scores = storage.scores.for_run(RUN_DATE)
    rankings = storage.rankings.for_run(RUN_DATE)

    naked = stored["outdoor"]
    naked.replace_tags([Tag(text="fireworks")])  # drops the vectors with them
    # Saved *before* the placements are put back, because `INSERT OR REPLACE
    # INTO events` is a delete and an insert: `event_scores` cascades away and
    # takes `rankings` with it. Written the other way round, this setup removed
    # the very event it meant to break and the rescore saw a healthy run.
    storage.events.save([naked])
    storage.scores.save(scores)
    storage.rankings.save(rankings)

    assert _rescore(db_path, _FixedWeather(_hour(66.0, "clear"))) is None

    after = {r.event_id: r for r in storage.rankings.for_run(RUN_DATE)}
    assert after["indoor"].rank == 1, "the stored ranking was overwritten"
    assert after["outdoor"].weather_adjustment == 0.0


def test_a_synthetic_activity_whose_rule_no_longer_fires_disappears(db_path):
    """Regenerated, not carried — which is the point of re-running enrichment.

    Synthetic rules are conditioned on the weather, so a refreshed forecast can
    make an evening walk stop qualifying. Carrying the stored row forward would
    keep it in the listing on a night it no longer belongs in, which is the
    class of staleness this whole path exists to remove.
    """
    walk = _event("synthetic:evening_walk:2026-08-17", "Evening walk", "outdoor", 0.45)
    walk.source_type = "synthetic"
    save_events([walk], db_path)

    storage = build_view_storage(db_path, "nomic-embed-text")
    scores = storage.scores.for_run(RUN_DATE)
    rankings = storage.rankings.for_run(RUN_DATE)
    storage.scores.save(
        [
            *scores,
            EventScore(
                event_id=walk.event_id,
                run_date=RUN_DATE,
                base_score=0.45,
                tag_confidence=1.0,
                match="yes",
                reasons=[],
            ),
        ]
    )
    storage.rankings.save(
        [
            *rankings,
            Ranking(
                event_id=walk.event_id,
                run_date=RUN_DATE,
                weather_adjustment=0.0,
                final_score=0.45,
                rank=3,
            ),
        ]
    )

    # No synthetic rules are configured, so enrichment regenerates none.
    refreshed = _rescore(db_path, _FixedWeather(_hour(66.0, "clear")))

    assert refreshed is not None
    assert walk.event_id not in [pair.event.event_id for pair in refreshed]
