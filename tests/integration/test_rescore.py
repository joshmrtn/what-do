"""The read-time rescore, end to end against real SQLite.

Everything real except the forecast, which is the one external boundary on this
path. In particular the pipeline stages are the real ones, so what these tests
constrain is what a `what-do` invocation actually does.
"""

from __future__ import annotations

import io
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
import json
from typing import Any

import pytest
import requests

from src.composition.storage import build_view_storage
from src.enrichment.air_quality import AIR_QUALITY_HOST
from src.enrichment.weather import OPEN_METEO_HOST, OpenMeteoProvider
from src.config import load_config
from src.config import (
    AppConfig,
    NetworkConfig,
    NetworkPolicy,
    ComfortCurve,
    LocationConfig,
    ScoringConfig,
    ScrapingConfig,
    SyntheticActivityRule,
    SyntheticConditions,
    VenueDiscoveryConfig,
    WeatherConfig,
)
from src.enrichment.air_quality import AIR_QUALITY_HOST
from src.enrichment.weather import OPEN_METEO_HOST
from src.models.event import Event
from src.models.event_score import EventScore
from src.models.ranking import Ranking
from src.models.tag import Tag
from src.composition.pipeline import RescoreUnavailable, run_view_tail
from src.presentation.rescore import rescore_if_stale
from src.scoring.embedding_stage import EmbeddingStage
from src.scoring.embeddings import EmbeddingError
from src.scoring.preferences import PreferenceSet
from src.scoring.ranking import RankingEngine
from src.scoring.similarity_stage import SimilarityStage
from src.storage.memory.events import InMemoryEventRepository
from src.storage.memory.rankings import InMemoryRankingRepository
from src.storage.memory.scores import InMemoryScoreRepository
from src.scoring.embedding_stage import embedding_input_hash
from src.scoring.embeddings import OllamaEmbeddingProvider
from src.scoring.similarity import Reason, SimilarityResult
from src.utils.vectors import encode_vector
from src.storage.events import save_events
from src.storage.queries import load_ranked_events
from src.storage.sqlite.connection import init_db
from src.utils.logging import get_logger
from src.utils.ollama_client import OllamaClient
from tests.support.network import fetcher_policy

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


class _CountingSession:
    """The transport, faked, counting what actually left the process."""

    def __init__(self) -> None:
        self.calls = 0

    def get(self, url: str, *, params=None, timeout=None):
        self.calls += 1
        hours = [f"{RUN_DATE.isoformat()}T{h:02d}:00" for h in range(24)]
        response = requests.Response()
        response.status_code = 200
        response._content = json.dumps(
            {
                "hourly": {
                    "time": hours,
                    "temperature_2m": [66.0] * 24,
                    "relative_humidity_2m": [40.0] * 24,
                    "dew_point_2m": [50.0] * 24,
                    "precipitation": [0.0] * 24,
                    "wind_speed_10m": [5.0] * 24,
                    "weather_code": [0] * 24,
                }
            }
        ).encode()
        return response


class _RefusingWeather:
    """Stands in for the network being down."""

    def fetch(self, day: date, lat: float, lng: float) -> dict[str, Any] | None:
        raise OSError("no route to host")


def _curve(ideal, zero, floor, weight=1.0, supersedes=()):
    return ComfortCurve(
        ideal=ideal, zero=zero, floor=floor, weight=weight, supersedes=supersedes
    )




def _network() -> NetworkConfig:
    """Declares the one host enrichment reads a cache lifetime for.

    There is deliberately no default policy, so a config that never mentions
    Open-Meteo is refused rather than guessed at — which is the behaviour, and
    means every config that enriches weather must say so.
    """
    return NetworkConfig(
        policies={
            "open_meteo": NetworkPolicy(
                min_interval_seconds=0.5,
                timeout_seconds=30.0,
                max_attempts=3,
                backoff_base_seconds=1.0,
                backoff_max_seconds=60.0,
                cache_ttl=timedelta(hours=12),
            )
        },
        hosts={OPEN_METEO_HOST: "open_meteo", AIR_QUALITY_HOST: "open_meteo"},
    )

def _config() -> AppConfig:
    return AppConfig(
        network=_network(),
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
    config: AppConfig | None = None,
) -> Any:
    storage = build_view_storage(db_path, "nomic-embed-text")
    pairs = load_ranked_events(storage.events, storage.scores, storage.rankings)
    return rescore_if_stale(
        pairs=pairs,
        tonight=tonight,
        now=now,
        ttl=TTL,
        config=config if config is not None else _config(),
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


def test_a_second_invocation_makes_no_request(db_path):
    """The forecast is cached, so the follow-up is instant.

    Asserted on requests rather than elapsed time, which would be a flaky way
    to say the same thing. The **real** provider is built here over a faked
    session: the cache lives at the request now, so a fake provider could not
    demonstrate this — it would simply answer twice.
    """
    session = _CountingSession()
    storage = build_view_storage(db_path, "nomic-embed-text")
    provider = OpenMeteoProvider(
        session=session,
        policy=fetcher_policy(
            urls=f"https://{OPEN_METEO_HOST}/v1/forecast", now=NOW
        ),
        weather_cache=storage.weather_cache,
        cache_ttl=timedelta(hours=12),
        get_now=lambda: NOW,
    )

    _rescore(db_path, provider)
    after_first = session.calls

    _rescore(db_path, provider, now=NOW + timedelta(minutes=5))

    assert after_first == 1
    assert session.calls == 1, "the second rescore refetched the forecast"


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


@pytest.mark.integration
def test_an_event_missing_a_vector_is_embedded_rather_than_refused(real_db_path):
    """Embedding at read time is allowed, so a gap is filled rather than fatal.

    This test previously asserted the opposite, because the read path held a
    provider that refused to embed anything. That enforced a rule that was never
    the rule: `CLAUDE.md` forbids **LLM** calls at query time, and an embedding
    is about a second. Corrected 2026-08-17.
    """
    storage = build_view_storage(real_db_path, "nomic-embed-text")
    stored = {event.event_id: event for event in storage.events.load_all()}
    scores = storage.scores.for_run(RUN_DATE)
    rankings = storage.rankings.for_run(RUN_DATE)

    naked = stored["outdoor"]
    naked.replace_tags([Tag(text="harbour fireworks")])  # drops the vectors
    # Saved before the placements are put back: `INSERT OR REPLACE INTO events`
    # is a delete and an insert, so `event_scores` cascades away and takes
    # `rankings` with it.
    storage.events.save([naked])
    storage.scores.save(scores)
    storage.rankings.save(rankings)

    refreshed = _rescore(real_db_path, _FixedWeather(_hour(66.0, "clear")))

    assert refreshed is not None, "an unembedded event abandoned the rescore"
    reloaded = {e.event_id: e for e in storage.events.load_all()}
    assert reloaded["outdoor"].tag_embeddings, "the missing vector was not computed"


def test_an_embedding_that_genuinely_fails_abandons_the_rescore():
    """The seal still exists; only its meaning changed.

    It no longer means "a model was refused" — it means the model could not be
    reached. `EmbeddingStage` degrades per event by design, so without this the
    event scores zero on its semantic half and that zero is written over a good
    stored score.
    """

    class _BrokenEmbedder:
        def embed(self, text: str) -> list[float]:
            raise EmbeddingError("ollama unreachable")

    event = _event("outdoor", "Harbour Fireworks", "outdoor", 0.50)
    event.replace_tags([Tag(text="fireworks")])  # drops the vectors with them

    with pytest.raises(RescoreUnavailable):
        run_view_tail(
            events=[event],
            run_date=RUN_DATE,
            now=NOW,
            config=_config(),
            enrichment_service=_PassThroughEnrichment(),
            embedding_stage=EmbeddingStage(
                _BrokenEmbedder(), get_logger("t", stream=io.StringIO())
            ),
            similarity_stage=SimilarityStage(PreferenceSet(), _config().scoring),
            ranking_engine=RankingEngine(_config()),
            event_repository=InMemoryEventRepository(),
            score_repository=InMemoryScoreRepository(),
            ranking_repository=InMemoryRankingRepository(),
            forecast_fresh_since=NOW - TTL,
        )


class _PassThroughEnrichment:
    """Leaves the events as they are, with their forecast already fresh.

    A boundary stand-in, not a reimplementation: this test is about the embedding
    seal, and real enrichment would reach the network to reach it.
    """

    def enrich(self, events: list[Event], run_date: date) -> list[Event]:
        for event in events:
            event.weather = {
                "sampled_hour": 20,
                "forecast": {
                    "issued_at": NOW.isoformat(),
                    "hour": _hour(66.0, "clear"),
                    "day_series": [],
                },
                "observed": None,
            }
        return events


def test_a_regenerated_event_reuses_the_vectors_of_an_identical_stored_one(db_path):
    """Synthetic text is authored, so tonight's copy is byte-identical.

    Embedding it would be a model call the read path may not make, and without
    the reuse a single evening walk abandoned every rescore — which is exactly
    the night a weather rescore is for.
    """
    from src.composition.pipeline import _carry_embeddings

    storage = build_view_storage(db_path, "nomic-embed-text")
    stored = storage.events.load_all()
    original = next(event for event in stored if event.event_id == "outdoor")

    # What enrichment hands back: same content, no vectors, no hash.
    regenerated = _event("outdoor", "Harbour Fireworks", "outdoor", 0.50)
    regenerated.replace_tags(list(original.tags))
    regenerated.replace_summary(original.summary)
    assert regenerated.tag_embeddings == []

    _carry_embeddings(stored, [regenerated])

    assert regenerated.tag_embeddings == original.tag_embeddings
    assert regenerated.summary_embedding == original.summary_embedding
    assert regenerated.embedding_input_hash == original.embedding_input_hash


def test_vectors_are_not_carried_onto_different_content(db_path):
    """Keyed on the text the vectors are a function of, never on the id.

    Carrying a vector onto changed text would score an event against something
    it does not say — silently, and with every downstream number looking fine.
    """
    from src.composition.pipeline import _carry_embeddings

    storage = build_view_storage(db_path, "nomic-embed-text")
    stored = storage.events.load_all()

    changed = _event("outdoor", "Harbour Fireworks", "outdoor", 0.50)
    changed.replace_tags([Tag(text="something else entirely")])

    _carry_embeddings(stored, [changed])

    assert changed.tag_embeddings == []


_WALK = SyntheticActivityRule(
    name="evening walk",
    conditions=SyntheticConditions(min_temp_f=55.0, weather=["clear"]),
    tags=["outdoor", "walking", "low_key"],
    summary="A pleasant evening walk",
    setting="outdoor",
)


def _with_walk() -> AppConfig:
    config = _config()
    config.synthetic_activities = [_WALK]
    return config


def _stored_walk() -> Event:
    """The walk as a completed batch left it, vectors and all."""
    walk = _event(f"synthetic:evening_walk:{RUN_DATE}", "Evening walk", "outdoor", 0.45)
    walk.source_type = "synthetic"
    walk.replace_tags([Tag(text=text) for text in _WALK.tags])
    walk.replace_summary(_WALK.summary)
    walk.attach_tag_embeddings([encode_vector([0.1, 0.2, 0.3])] * len(_WALK.tags))
    walk.summary_embedding = encode_vector([0.4, 0.5, 0.6])
    walk.embedding_input_hash = embedding_input_hash(walk)
    return walk


def test_a_regenerated_synthetic_does_not_abandon_the_rescore(db_path):
    """The night a weather rescore is *for* is the night a walk qualifies.

    Enrichment mints the walk fresh, with authored text and no vectors, and
    embedding it is a model call the read path may not make. Without the reuse
    this one event abandoned every rescore — which is what the live run showed.
    """
    walk = _stored_walk()
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

    refreshed = _rescore(
        db_path, _FixedWeather(_hour(66.0, "clear")), config=_with_walk()
    )

    assert refreshed is not None, "a regenerated walk abandoned the rescore"
    assert walk.event_id in [pair.event.event_id for pair in refreshed]


def _real_embedder():
    """The provider the batch uses, for a test that needs real vectors.

    Fake vectors cannot serve here. Preference lines are embedded for real by
    the read path, so an event carrying a hand-written 3-element vector fails on
    `Cannot compare vectors of different length: 3 != 768` — which is how the
    first cut of these tests failed, in the fixture rather than the code.
    """
    config = load_config(Path("config/config.yaml"))
    return OllamaEmbeddingProvider(
        OllamaClient(
            config.ollama_host,
            timeout=config.models.request_timeout_seconds,
            component="embedding",
        ),
        model=config.models.embeddings,
    )


def _really_embedded(event: Event, embedder) -> Event:
    """Give an event vectors from the real model, as a completed batch would."""
    event.attach_tag_embeddings(
        [encode_vector(embedder.embed(tag.text)) for tag in event.tags]
    )
    if event.summary:
        event.summary_embedding = encode_vector(embedder.embed(event.summary))
    event.embedding_input_hash = embedding_input_hash(event)
    return event


@pytest.fixture
def real_db_path(tmp_path: Path) -> Path:
    """The same two events, embedded by the real model."""
    embedder = _real_embedder()
    path = tmp_path / "event_hub.db"
    init_db(path)

    outdoor = _really_embedded(
        _event("outdoor", "Harbour Fireworks", "outdoor", 0.50), embedder
    )
    outdoor.replace_tags([Tag(text="fireworks")])
    outdoor.replace_summary("An outdoor fireworks display over the harbour")
    _really_embedded(outdoor, embedder)

    indoor = _really_embedded(
        _event("indoor", "Cellar Jazz", "indoor", 0.52), embedder
    )
    indoor.replace_tags([Tag(text="jazz")])
    indoor.replace_summary("A live jazz quartet in a basement club")
    _really_embedded(indoor, embedder)

    save_events([outdoor, indoor], path)

    storage = build_view_storage(path, "nomic-embed-text")
    storage.scores.save(
        [
            EventScore(
                event_id=event_id,
                run_date=RUN_DATE,
                base_score=base,
                tag_confidence=1.0,
                match="yes",
                reasons=[],
            )
            for event_id, base in (("outdoor", 0.50), ("indoor", 0.52))
        ]
    )
    storage.rankings.save(
        [
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
    )
    return path


def _rescore_with_preferences(db_path: Path, tmp_path: Path, likes: str, dislikes: str):
    likes_path = tmp_path / "likes.txt"
    dislikes_path = tmp_path / "dislikes.txt"
    likes_path.write_text(likes)
    dislikes_path.write_text(dislikes)

    storage = build_view_storage(db_path, "nomic-embed-text")
    refreshed = rescore_if_stale(
        pairs=load_ranked_events(storage.events, storage.scores, storage.rankings),
        tonight=RUN_DATE,
        now=NOW,
        ttl=TTL,
        config=_config(),
        db_path=db_path,
        storage=storage,
        logger=get_logger("rescore_test", stream=io.StringIO()),
        get_now=lambda: NOW,
        weather_provider=_FixedWeather(_hour(66.0, "clear")),
        likes_path=likes_path,
        dislikes_path=dislikes_path,
        blocklist_path=tmp_path / "blocklist.json",
    )
    return refreshed, storage


@pytest.mark.integration
def test_an_edited_preference_file_rescores_at_read_time(real_db_path, tmp_path):
    """The feature this path exists for, alongside the forecast.

    Editing `likes.txt` and running `what-do` must show the new ordering rather
    than waiting for the overnight batch. A changed line has no cached vector, so
    this embeds one — about a second, which is the whole argument. Reading the
    "no LLM at query time" rule as a ban on embeddings was an over-tightening,
    corrected 2026-08-17: the rule names LLM calls and nothing else.
    """
    refreshed, storage = _rescore_with_preferences(
        real_db_path, tmp_path, "live jazz in a basement club\n", "fireworks\n"
    )

    assert refreshed is not None, "the rescore was abandoned over a preference edit"
    scores = {s.event_id: s for s in storage.scores.for_run(RUN_DATE)}
    assert scores["indoor"].base_score > scores["outdoor"].base_score, (
        "the semantic half did not move, so the preference edit did not reach scoring"
    )


@pytest.mark.integration
def test_reversing_the_preferences_reverses_the_order(real_db_path, tmp_path):
    """The mutation-proof for the test above: the same machinery, opposite input.

    Without it, "indoor scores higher" could be a property of the fixture rather
    than of the preferences that were just typed.
    """
    refreshed, storage = _rescore_with_preferences(
        real_db_path, tmp_path, "outdoor fireworks over the water\n", "jazz\n"
    )

    assert refreshed is not None
    scores = {s.event_id: s for s in storage.scores.for_run(RUN_DATE)}
    assert scores["outdoor"].base_score > scores["indoor"].base_score


@pytest.mark.integration
def test_the_preferences_a_rescore_used_are_recorded(real_db_path, tmp_path):
    """So `--explain` can attribute the numbers to a preference set."""
    _, storage = _rescore_with_preferences(
        real_db_path, tmp_path, "live jazz\n", "fireworks\n"
    )

    recorded = storage.rescores.latest_for(RUN_DATE)

    assert recorded is not None
    assert recorded.preference_revision_id is not None
