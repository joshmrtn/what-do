"""Unit tests for the batch orchestrator's sequencing and save points."""

from __future__ import annotations

import copy
import hashlib
import io
import json
import sqlite3
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pytest
from unittest.mock import MagicMock

from src.config import (
    AppConfig,
    DeduplicationConfig,
    LocationConfig,
    ScrapingConfig,
    VenueDiscoveryConfig,
)
from src.ingestion.ingestion_service import IngestionService, SourceTally
from src.ingestion.source import IngestionSource
from src.models.event import Event
from src.models.event_candidate import EventCandidate
from src.models.tag import Tag
from src.normalization.service import NormalizationService
from src.enrichment.service import EnrichmentService
from src.normalization.semantic_dedup import SemanticDeduplicationEngine
from src.processing.extraction import ExtractionResult
from src.storage.sqlite.connection import connect
from src.storage.sqlite.curve_state import SqliteCurveStateRepository
from src.storage.sqlite.dedup_decisions import SqliteDedupDecisionRepository
from src.storage.sqlite.extraction_observations import (
    SqliteExtractionObservationRepository,
)
from src.processing.extraction_stage import ExtractionStage, extraction_input_hash
from src.scheduler import run_batch
from src.scoring.embedding_stage import EmbeddingStage
from src.scoring.preferences import PreferenceSet
from src.scoring.ranking import RankingEngine
from src.scoring.similarity_stage import SimilarityStage
from src.enrichment.astronomical import AstronomicalCalculator
from src.enrichment.weather import WeatherProvider
from src.storage.memory.weather_cache import InMemoryWeatherCache
from src.storage.sqlite.connection import init_db
from src.storage.events import load_events, save_events
from src.models.event_score import EventScore
from src.models.ranking import Ranking
from src.storage.memory.candidates import InMemoryCandidateRepository
from src.storage.memory.entities import InMemoryEntityRepository
from src.storage.memory.rankings import InMemoryRankingRepository
from src.storage.memory.scores import InMemoryScoreRepository
from src.storage.sqlite.events import SqliteEventRepository
from src.storage.sqlite.runs import SqliteRunRepository
from src.utils.logging import get_logger

NOW = datetime(2026, 6, 15, 12, 0, 0, tzinfo=timezone.utc)
RUN_DATE = date(2026, 6, 15)


# ----------------------------------------------------------------------
# Fakes
# ----------------------------------------------------------------------


class _IngestionSpy:
    """Records how ingestion was called, and delegates to the real service.

    `persist` and `collect_raw` used to be recorded by a fake that then ignored
    them, so a dry run was tested by asserting a flag had been *passed* rather
    than *honoured*.
    """

    def __init__(self, inner, error: Exception | None = None) -> None:
        self._inner = inner
        self.error = error
        self.calls = 0
        self.persist_flags: list[bool] = []
        self.raw_flags: list[bool] = []

    def run(self, get_now=None, persist=True, collect_raw=False):
        self.calls += 1
        self.persist_flags.append(persist)
        self.raw_flags.append(collect_raw)
        if self.error:
            raise self.error
        return self._inner.run(
            get_now=get_now, persist=persist, collect_raw=collect_raw
        )


class _NormalizationSpy:
    """Records the candidates normalization saw, and delegates to the real one."""

    def __init__(self, inner) -> None:
        self._inner = inner
        self.calls: list[list[EventCandidate]] = []

    def run(self, candidates, get_now=None):
        self.calls.append(list(candidates))
        return self._inner.run(candidates, get_now=get_now)


class _EnrichmentSpy:
    """Records what enrichment saw, and delegates to the real service."""

    def __init__(self, inner, error: Exception | None = None) -> None:
        self._inner = inner
        self.error = error
        self.seen: list[list[str]] = []
        self.titles: list[list[str]] = []

    def enrich(self, events, run_date):
        if self.error:
            raise self.error
        self.seen.append([e.event_id for e in events])
        self.titles.append([e.title for e in events])
        return self._inner.enrich(events, run_date)


class _ExtractionModel:
    """The model boundary — the only seam an extraction test may substitute.

    Records every call, which is how a test observes what the *real*
    `ExtractionStage` decided to extract. A fake stage cannot serve that
    purpose: it would restate the skip rule rather than exercise it, and the one
    it restated ("skip any event that already has tags") was never the rule
    production runs, which is on `extraction_input_hash`.
    """

    def __init__(self) -> None:
        self.calls: list[str] = []

    def extract(self, text, image_bytes=None, reference_date=None):
        self.calls.append(text)
        return ExtractionResult(
            title=None,
            venue=None,
            start_time=None,
            end_time=None,
            tags=[Tag(text="karaoke", weight=1.0)],
            # Derived from the input, because a real model's is. A constant
            # summary gives every event an identical one, and dedup pass 2 —
            # now real — merges them as duplicates of each other.
            summary=f"Karaoke night: {text}",
            model="fake-extraction-model",
            prompt_version="fakever0",
            degradation=None,
            setting="indoor",
        )


class _EmbeddingModel:
    """The model boundary for embeddings. Same rule, same reason."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def embed(self, text: str) -> list[float]:
        """Deterministic, and different for different text.

        A constant vector makes every event a perfect semantic duplicate of
        every other, which the real dedup engine then acts on.
        """
        self.calls.append(text)
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        return [byte / 255.0 for byte in digest[:8]]


class _StageSpy:
    """Records that a stage ran, and delegates the work to the real one.

    The sanctioned kind of double: it records, it does not reimplement, so it
    cannot drift from the thing it wraps.
    """

    def __init__(self, inner, error: Exception | None = None) -> None:
        self._inner = inner
        self.error = error
        self.runs = 0
        self.save_fn = None
        self.scope_fn = None

    def set_save_fn(self, save_fn):
        self.save_fn = save_fn
        self._inner.set_save_fn(save_fn)

    def set_scope_fn(self, scope_fn):
        self.scope_fn = scope_fn
        self._inner.set_scope_fn(scope_fn)

    def process(self, events):
        self.runs += 1
        if self.error:
            raise self.error
        return self._inner.process(events)

    def __getattr__(self, name):
        """Everything not recorded here belongs to the stage being wrapped.

        Without this the spy silently *narrows* the interface it stands for —
        `extraction_stage.deferred` raised `AttributeError` through the spy
        while working perfectly in production, which is the drift a recording
        double is supposed to be incapable of.
        """
        return getattr(self._inner, name)


def _stage_log():
    return get_logger("batch_test_stages", stream=io.StringIO())


def _extraction_stage(error: Exception | None = None) -> _StageSpy:
    """The real stage behind a recording spy, with only the model substituted."""
    return _StageSpy(
        ExtractionStage(_ExtractionModel(), None, _stage_log(), get_now=lambda: NOW),
        error=error,
    )


def _embedding_stage(error: Exception | None = None) -> _StageSpy:
    return _StageSpy(EmbeddingStage(_EmbeddingModel(), _stage_log()), error=error)


def _enrichment_service(db, error: Exception | None = None) -> _EnrichmentSpy:
    """The real service, with only its external providers substituted.

    Nine constructor dependencies is the point of injecting them, not an
    argument against building the real thing: five are external providers that
    already have doubles elsewhere, and the rest are config and storage that a
    test has anyway.
    """
    # Open-Meteo is the only genuine boundary here. `fetch` returns a day as
    # `{"date", "hours"}` — the shape a flat daily dict was silently swapped for
    # on 2026-08-04 while 650 tests stayed green, so it is written out rather
    # than left to a bare MagicMock that would accept anything.
    weather = MagicMock(spec=WeatherProvider)
    weather.fetch.return_value = {
        "date": RUN_DATE.isoformat(),
        "hours": [
            {"hour": h, "temperature_f": 68.0, "precipitation_probability": 0,
             "wind_speed_mph": 4.0, "condition": "clear"}
            for h in range(24)
        ],
    }
    return _EnrichmentSpy(
        EnrichmentService(
            weather_provider=weather,
            movie_provider=None,
            # Pure — astral, no I/O — so there is nothing here to substitute.
            astronomical_calculator=AstronomicalCalculator(),
            synthetic_rules=[],
            config=_config(),
            db_path=db,
            weather_cache=InMemoryWeatherCache(),
            get_now=lambda: NOW,
            logger=_stage_log(),
        ),
        error=error,
    )


def _sourced(name: str, candidates):
    """A source standing in for the network, named as config would name it.

    `candidates` may be an exception, for a source that fails while its
    siblings succeed — which the real service reports rather than raising.
    """
    source = MagicMock(spec=IngestionSource)
    source.source_name = name
    if isinstance(candidates, Exception):
        source.fetch.side_effect = candidates
    else:
        source.fetch.return_value = list(candidates)
    return source


def _ingestion_service(db, candidates=None, sources=None, error: Exception | None = None):
    """The real service, fed by a source that stands in for the network.

    A seeds file and an entity store are all the setup this needs — the file is
    two lines and the store is the in-memory repository that already exists —
    which is a smaller price than a fake we have to remember to update every
    time ingestion grows.
    """
    seeds = Path(db).parent / "seeds.yaml"
    seeds.write_text("handles: []\nvenues: []\n")
    independent = (
        [_sourced(name, cands) for name, cands in sources.items()]
        if sources is not None
        else [_sourced("test_source", candidates or [])]
    )
    return _IngestionSpy(
        IngestionService(
            config=_config(),
            db_path=db,
            seeds_path=seeds,
            failover_sources=[],
            independent_sources=independent,
            logger=_stage_log(),
            entities=InMemoryEntityRepository(),
        ),
        error=error,
    )


def _normalization_service() -> _NormalizationSpy:
    return _NormalizationSpy(NormalizationService(_config(), _stage_log()))


def _similarity_stage() -> SimilarityStage:
    """Real, over an empty preference set — the scheduler tests are about
    sequencing, and an empty set is a legitimate configuration, not a stand-in."""
    return SimilarityStage(PreferenceSet(), _config().scoring)


def _ranking_engine() -> _RankingSpy:
    return _RankingSpy(RankingEngine(_config(), logger=_stage_log()))


class _DedupSpy:
    """Records that dedup pass 2 ran, and delegates to the real engine.

    The engine is pure and takes no constructor arguments at all, so there was
    never a cost that justified restating it.
    """

    def __init__(self, inner) -> None:
        self._inner = inner
        self.calls = 0

    def deduplicate(self, events, config, **kwargs):
        self.calls += 1
        # Forwarded rather than restated: a recording double that pins the
        # signature narrows the interface it stands for, and a spy that cannot
        # pass an argument through has started reimplementing.
        return self._inner.deduplicate(events, config, **kwargs)


class _RankingSpy:
    """Records what was ranked, and delegates to the real engine.

    The old fake invented scores — a `base_score` of 1.0 and a `match` of "yes"
    for every event, in rank order of arrival. Nothing asserted on those
    numbers, but the run persisted them, so every downstream assertion about
    scores and rankings was reading fiction.
    """

    def __init__(self, inner) -> None:
        self._inner = inner
        self.ranked: list[list[str]] = []
        self.titles: list[list[str]] = []

    def rank(self, events, run_date):
        # Ids are uuids minted per run, so a test that cares *which* event was
        # ranked reads titles; one that cares about identity surviving a run
        # compares ids across two runs, which is the thing reconcile exists for.
        self.ranked.append([e.event_id for e in events])
        self.titles.append([e.title for e in events])
        return self._inner.rank(events, run_date)


class _Spy:
    """Wraps a storage function and records a snapshot of every call."""

    def __init__(self, fn) -> None:
        self._fn = fn
        self.snapshots: list[list[tuple[str, bool, bool]]] = []

    def __call__(self, items, db_path):
        if items and isinstance(items[0], Event):
            self.snapshots.append(
                [(e.event_id, bool(e.tags), bool(e.tag_embeddings)) for e in items]
            )
        else:
            self.snapshots.append(list(items))
        return self._fn(items, db_path)

    @property
    def calls(self) -> int:
        return len(self.snapshots)


class _SpyRankingRepository:
    """An in-memory ranking repository that records every write.

    Replaces the old save-function spy: rankings now go through a repository,
    and the in-memory one is the official fake.
    """

    def __init__(self) -> None:
        self._inner = InMemoryRankingRepository()
        self.calls = 0
        self.snapshots: list[list[Ranking]] = []

    def save(self, rankings: list[Ranking]) -> None:
        self.calls += 1
        self.snapshots.append(list(rankings))
        self._inner.save(rankings)

    def for_run(self, run_date):
        return self._inner.for_run(run_date)

    def latest_run_date(self):
        return self._inner.latest_run_date()


class _SpyRepository:
    """A real repository that records every write.

    Delegates rather than reimplements: the tests below assert on what the
    batch persisted, and a hand-written store would only prove the batch agrees
    with the stub. `on_save` replaces the write entirely, for the runs that need
    persistence to fail.
    """

    def __init__(self, inner, on_save=None, on_replace=None) -> None:
        self._inner = inner
        self._on_save = on_save
        self._on_replace = on_replace
        self.snapshots: list[list[tuple[str, bool, bool]]] = []
        self.replaced: list[list[str]] = []

    def _record(self, events) -> None:
        self.snapshots.append(
            [(e.title, bool(e.tags), bool(e.tag_embeddings)) for e in events]
        )

    def save(self, events) -> None:
        self._record(events)
        if self._on_save is not None:
            self._on_save(events)
            return
        self._inner.save(events)

    def save_one(self, event) -> None:
        self._record([event])
        if self._on_save is not None:
            self._on_save([event])
            return
        self._inner.save_one(event)

    def replace(self, stale_ids, events) -> None:
        self._record(events)
        self.replaced.append(list(stale_ids))
        if self._on_replace is not None:
            self._on_replace(events)
            return
        if self._on_save is not None:
            self._on_save(events)
            return
        self._inner.replace(stale_ids, events)

    def load_all(self):
        return self._inner.load_all()

    def delete(self, event_ids) -> None:
        self._inner.delete(event_ids)

    def tag_embeddings(self):
        return self._inner.tag_embeddings()

    @property
    def calls(self) -> int:
        return len(self.snapshots)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _config(**scraping) -> AppConfig:
    return AppConfig(
        location=LocationConfig(42.52, -70.89, "01970", 10.0, "America/New_York"),
        scraping=ScrapingConfig(**scraping),
        venue_discovery=VenueDiscoveryConfig(),
        # Sampling off. These tests ask whether a decision reaches the table at
        # all, and under the shipped 1-in-10 rule a single below-floor pair is
        # dropped nine times out of ten — which would make a wiring test fail
        # on a coin flip. What gets sampled is `test_decision_sampling.py`'s
        # question, and it answers it deterministically.
        deduplication=DeduplicationConfig(decision_sample_denominator=1),
    )


def _event(event_id: str, candidates: list[str], **overrides) -> Event:
    fields = {
        "event_id": event_id,
        "source_event_candidates": candidates,
        "source_type": "apify",
        "created_at": NOW,
        "updated_at": NOW,
        "title": "Karaoke Night",
        "start_time": NOW + timedelta(days=1),
    }
    fields.update(overrides)
    return Event(**fields)


def _candidate(cid: str, title: str = "Karaoke Night", **overrides) -> EventCandidate:
    """A candidate as a source would hand it over, before normalization."""
    fields = {
        "id": cid,
        "source": "@venue",
        "source_type": "apify",
        "discovered_at": NOW,
        "title": title,
        "start_time": NOW + timedelta(days=1),
    }
    fields.update(overrides)
    return EventCandidate(**fields)


@pytest.fixture
def db(tmp_path) -> Path:
    path = tmp_path / "batch.db"
    init_db(path)
    return path


def _seeded_candidates(candidates) -> InMemoryCandidateRepository:
    """The candidates a previous fetch left behind.

    Mirrors what the sources return, because in production ingestion
    persists what it fetched and the next stage reads it back. Seeding a
    candidate no source returns makes every run process a ghost.
    """
    repo = InMemoryCandidateRepository()
    # A complete candidate, as a real fetch would have left it. A bare one
    # (no title, no start_time) is discarded by real normalization, and
    # `_merge_candidates` prefers the *stored* copy over the fetched one.
    repo.save(list(candidates))
    return repo


def _run(db, *, candidates=None, stored_candidates=None, deps=None, **kwargs):
    """Drive run_batch, returning (result, the collaborators it used).

    Everything here is the real thing except the network and the model: one
    `IngestionSource` supplying candidates, one extraction model and one
    embedding model. Events therefore carry the uuids normalization mints,
    exactly as they do in production, which is why assertions below identify
    them by title rather than by a name a fixture chose.
    """
    candidates = [_candidate("c1")] if candidates is None else candidates
    # What a previous run left in the repository. Defaults to what the
    # sources return, because ingestion persists what it accepted — so a
    # candidate ingestion *rejects* must be passed as [] here, or the
    # loader hands the pipeline something production never stored.
    stored = candidates if stored_candidates is None else stored_candidates
    log = get_logger("batch_test_stages", stream=io.StringIO())
    extraction_model = _ExtractionModel()
    embedding_model = _EmbeddingModel()
    fakes = {
        "ingestion_service": _ingestion_service(db, candidates),
        "normalization_service": _normalization_service(),
        "enrichment_service": _enrichment_service(db),
        # Real stages, faked models. The seam is the model call, which is the
        # only external thing here; everything above it is our code and runs
        # for real, so these tests constrain what production actually does.
        "extraction_stage": _StageSpy(
            ExtractionStage(extraction_model, None, log, get_now=lambda: NOW)
        ),
        "embedding_stage": _StageSpy(EmbeddingStage(embedding_model, log)),
        "semantic_deduplicator": _DedupSpy(SemanticDeduplicationEngine()),
        "similarity_stage": _similarity_stage(),
        "ranking_engine": _ranking_engine(),
    }
    fakes.update(deps or {})
    # Kept out of the kwargs run_batch receives — it takes stages, not models.
    fakes["extraction_model"] = extraction_model
    fakes["embedding_model"] = embedding_model
    stages = {k: v for k, v in fakes.items() if not k.endswith("_model")}

    # A test that needs to inspect the store afterwards supplies its own, so it
    # holds the same object the batch reads through.
    candidates_repo = kwargs.pop("candidate_repository", None) or _seeded_candidates(stored)

    save_spy = _SpyRepository(
        SqliteEventRepository(db),
        on_save=kwargs.pop("on_save", None),
        on_replace=kwargs.pop("on_replace", None),
    )
    recs_spy = _SpyRankingRepository()

    # Real repositories against the test database, so the refit and its
    # observation log run on every batch here rather than only where a test
    # thought to ask for them. `setdefault` leaves a test free to supply its own.
    kwargs.setdefault("curve_state_repository", SqliteCurveStateRepository(db))
    kwargs.setdefault(
        "extraction_observation_repository", SqliteExtractionObservationRepository(db)
    )

    result = run_batch(
        config=_config(),
        db_path=db,
        logger=get_logger("batch_test", stream=io.StringIO()),
        get_now=lambda: NOW,
        run_date=RUN_DATE,
        candidate_repository=candidates_repo,
        run_repository=SqliteRunRepository(db),
        event_repository=save_spy,
        score_repository=InMemoryScoreRepository(),
        ranking_repository=recs_spy,
        dedup_decision_repository=SqliteDedupDecisionRepository(db),
        **stages,
        **kwargs,
    )
    return result, fakes, save_spy, recs_spy


# ----------------------------------------------------------------------
# Sequence
# ----------------------------------------------------------------------


def test_a_clean_run_reports_success(db):
    result, _, _, _ = _run(db)
    assert result.outcome == "success"
    assert result.errors == []


def test_ingestion_runs_before_candidates_are_read(db):
    result, fakes, _, _ = _run(db)
    assert fakes["ingestion_service"].calls == 1


def test_skip_ingest_leaves_the_network_alone(db):
    result, fakes, _, _ = _run(db, skip_ingest=True)
    assert fakes["ingestion_service"].calls == 0
    assert result.outcome == "success"


def test_recommendations_are_saved(db):
    _, _, _, recs_spy = _run(db)
    assert recs_spy.calls == 1
    assert len(recs_spy.snapshots[0]) == 1


# ----------------------------------------------------------------------
# Save points
# ----------------------------------------------------------------------


def test_events_are_saved_after_extraction(db):
    """Three minutes an event of LLM time must survive a crash in a later stage."""
    _, _, save_spy, _ = _run(db)

    first = save_spy.snapshots[0]
    assert first == [("Karaoke Night", True, False)]


def test_events_are_saved_after_embedding(db):
    _, _, save_spy, _ = _run(db)

    # The last write, not the second: the real stage checkpoints after every
    # event, so the run makes more saves than the old stage fake did — it held
    # the save function and never called it, leaving the checkpoint path
    # unexercised by every test in this module.
    assert ("Karaoke Night", True, True) in save_spy.snapshots[-1]


def test_a_crash_after_extraction_leaves_tags_persisted(db):
    """Everything already saved stays saved, which is what makes a re-run cheap."""
    _run(db, deps={"embedding_stage": _embedding_stage(error=RuntimeError("ollama down"))})

    stored = load_events(db)
    assert [t.text for t in stored[0].tags] == ["karaoke"]


def test_a_second_run_re_extracts_nothing(db):
    """The whole point of persisting: model time is paid once per event."""
    _run(db)
    _, fakes, _, _ = _run(db)

    assert fakes["extraction_model"].calls == []


def test_a_second_run_reuses_the_stored_event_id(db):
    """Normalization mints a new uuid every run; reconcile matches it back.

    This test could not fail while normalization was faked. The fake deep-copied
    a fixture, so the second run handed back the very id the first one stored —
    the assertion was satisfied by the double, not by the code. Neutering
    reconcile left it green.

    Now the two ids are genuinely different objects until reconcile matches them
    on their shared candidate, which is the mechanism that stops the events
    table doubling every night.
    """
    _, first, _, _ = _run(db)
    _, second, _, _ = _run(db)

    minted_first = first["ranking_engine"].ranked[0]
    minted_second = second["ranking_engine"].ranked[0]

    assert minted_first == minted_second
    assert len(load_events(db)) == 1


# ----------------------------------------------------------------------
# Reconcile wiring
# ----------------------------------------------------------------------


def test_stored_enrichment_is_adopted_by_the_matching_fresh_event(db):
    stored = _event(
        "stored-1",
        ["c1"],
        tags=[Tag(text="trivia", weight=1.0)],
        summary="Trivia night",
        setting="indoor",
    )
    # A stored event that has been through extraction carries the hash of what
    # was extracted; that hash, not the presence of tags, is what makes the next
    # run skip it. The fixture used to omit it and the stage fake did not care,
    # so this asserted a skip that production would never have made.
    stored.extraction_input_hash = extraction_input_hash(stored)
    save_events([stored], db)

    _, fakes, _, _ = _run(db)

    assert fakes["ranking_engine"].titles == [["Karaoke Night"]]
    assert fakes["extraction_model"].calls == []


def test_a_superseded_event_leaves_the_working_set_but_not_the_database(db):
    """Was `test_a_superseded_event_is_deleted`, which asserted the delete
    itself. The requirement it was protecting — the loser must not linger as a
    duplicate in the output — is unchanged; what satisfies it is not. The row
    is kept as half of a labelled cluster, and the repository's default filter
    is what keeps it out of the working set.
    """
    save_events([_event("stored-a", ["c1"]), _event("stored-b", ["c2"])], db)

    _run(db, candidates=[_candidate("c1"), _candidate("c2")])

    live = SqliteEventRepository(db).load_all()
    everything = SqliteEventRepository(db).load_all(include_superseded=True)

    assert [e.event_id for e in live] == ["stored-a"], "the loser is out of the way"
    assert sorted(e.event_id for e in everything) == ["stored-a", "stored-b"]


def test_a_stored_event_with_no_fresh_counterpart_is_still_ranked(db):
    """Its candidates aged out of the window; the event itself has not.

    The title has to differ from the fresh event's. Both fixtures used the
    default, which made them word-for-word identical, and dedup pass 2 — real
    here now — correctly merged them. The old fake merged nothing, so the
    fixture could get away with describing two copies of one event.
    """
    save_events([_event("stored-old", ["c99"], title="Quiz Night")], db)

    _, fakes, _, _ = _run(db)

    assert sorted(fakes["ranking_engine"].titles[0]) == ["Karaoke Night", "Quiz Night"]


# ----------------------------------------------------------------------
# Failure policy
# ----------------------------------------------------------------------


def test_ingestion_failure_does_not_abort_the_run(db):
    """A night where every source is down should still answer from what we have."""
    save_events([_event("stored-1", ["c1"])], db)

    result, fakes, _, recs_spy = _run(
        db,
        candidates=[],
        deps={"ingestion_service": _ingestion_service(db, error=RuntimeError("all sources down"))},
    )

    assert result.outcome == "partial"
    assert fakes["ranking_engine"].titles == [["Karaoke Night"]]
    assert recs_spy.calls == 1


def test_a_stage_failure_is_recorded_in_the_result(db):
    result, _, _, _ = _run(
        db, deps={"enrichment_service": _enrichment_service(db, error=RuntimeError("weather down"))}
    )

    assert result.outcome == "partial"
    assert any("weather down" in e for e in result.errors)


def test_enrichment_failure_still_reaches_ranking(db):
    _, fakes, _, _ = _run(
        db, deps={"enrichment_service": _enrichment_service(db, error=RuntimeError("weather down"))}
    )

    assert fakes["ranking_engine"].titles == [["Karaoke Night"]]


def test_wholesale_embedding_failure_stops_before_ranking(db):
    """Ranking without vectors produces garbage, and it would be persisted."""
    result, fakes, _, recs_spy = _run(
        db, deps={"embedding_stage": _embedding_stage(error=RuntimeError("ollama down"))}
    )

    assert fakes["ranking_engine"].ranked == []
    assert recs_spy.calls == 0
    assert result.outcome == "failed"


# ----------------------------------------------------------------------
# Ranking scope
# ----------------------------------------------------------------------


def _scope_run(db, event):
    """Store one event and run with no fresh candidates.

    `_scope_filter` is the scheduler's own predicate and it runs over carried
    forward *stored* events. Feeding these in as fresh candidates would have
    them dropped by ingestion's window check first — the same verdict for a
    different reason, one layer too early, leaving this filter untested.
    """
    save_events([event], db)
    _, fakes, _, _ = _run(db, candidates=[])
    return fakes["ranking_engine"].titles[0]


def test_events_beyond_the_horizon_are_not_ranked(db):
    """Beyond whatever the horizon is, rather than beyond a number written here
    twice. Pinned at 90 this read as a boundary test and was in fact an equality
    test against the default, so raising the default broke it."""
    beyond = timedelta(days=ScrapingConfig().horizon_days + 30)
    far = _event("far", ["c1"], title="Far", start_time=NOW + beyond)

    assert _scope_run(db, far) == []


def test_events_already_past_are_not_ranked(db):
    gone = _event("gone", ["c1"], title="Gone", start_time=NOW - timedelta(days=2))

    assert _scope_run(db, gone) == []


def test_undated_events_inside_the_lookback_are_ranked(db):
    """The CLI ranks them inline and must not lose them."""
    undated = _event("undated", ["c1"], title="Undated", start_time=None)

    assert _scope_run(db, undated) == ["Undated"]


def test_undated_events_older_than_the_lookback_are_not_ranked(db):
    stale = _event(
        "stale", ["c1"], title="Stale", start_time=None,
        created_at=NOW - timedelta(days=200),
    )

    assert _scope_run(db, stale) == []


def test_an_event_tonight_is_ranked(db):
    tonight = _event("tonight", ["c1"], title="Tonight", start_time=NOW + timedelta(hours=8))

    assert _scope_run(db, tonight) == ["Tonight"]


# ----------------------------------------------------------------------
# Ingest-only
# ----------------------------------------------------------------------


def test_ingest_only_stops_before_the_pipeline(db):
    """It exists to prove the sources fetch, without paying for extraction."""
    _, fakes, _, _ = _run(db, ingest_only=True)

    assert fakes["ingestion_service"].calls == 1
    assert fakes["normalization_service"].calls == []
    assert fakes["extraction_stage"].runs == 0
    assert fakes["ranking_engine"].ranked == []


def test_ingest_only_persists_the_candidates_it_fetched(db):
    """So a later --skip-ingest run can process them without refetching."""
    _, fakes, _, _ = _run(db, ingest_only=True)

    assert fakes["ingestion_service"].persist_flags == [True]


def test_ingest_only_with_dry_run_persists_nothing(db):
    _, fakes, _, _ = _run(db, ingest_only=True, dry_run=True)

    assert fakes["ingestion_service"].persist_flags == [False]


def test_ingest_only_writes_no_events_or_recommendations(db):
    _, _, save_spy, recs_spy = _run(db, ingest_only=True)

    assert (save_spy.calls, recs_spy.calls) == (0, 0)


def test_ingest_only_reports_the_count_it_accepted(db):
    ingestion = _ingestion_service(db, [_candidate("a"), _candidate("b", title="Quiz")])

    result, _, _, _ = _run(db, ingest_only=True, deps={"ingestion_service": ingestion})

    assert result.stage_counts["ingested"] == 2


def test_ingest_only_reports_what_each_source_returned(db):
    """A total says nothing about which of seventeen sources went quiet."""
    ingestion = _ingestion_service(
        db,
        sources={
            "do617_gulu_gulu": [_candidate(f"g{i}", title=f"Gig {i}") for i in range(25)],
            "do617_koto": [],
        },
    )

    result, _, _, _ = _run(db, ingest_only=True, deps={"ingestion_service": ingestion})

    # `SourceTally`, not an int: both numbers are needed to tell "returned
    # nothing" from "returned plenty and kept none". The old fake handed back
    # bare ints, so this asserted a shape the service never produced.
    assert result.per_source == {
        "do617_gulu_gulu": SourceTally(fetched=25, accepted=25),
        "do617_koto": SourceTally(fetched=0, accepted=0),
    }


def test_ingest_only_reports_a_source_that_failed(db):
    ingestion = _ingestion_service(
        db, sources={"nshoremag": RuntimeError("site down")}
    )

    result, _, _, _ = _run(db, ingest_only=True, deps={"ingestion_service": ingestion})

    assert result.failed_sources == ["nshoremag"]


def test_ingest_only_records_no_run_history(db):
    """A diagnostic did not do a batch, as with a dry run."""
    _run(db, ingest_only=True)

    assert _runs(db) == []


def test_ingest_only_survives_an_ingestion_failure(db):
    ingestion = _ingestion_service(db, error=RuntimeError("network down"))

    result, _, _, _ = _run(db, ingest_only=True, deps={"ingestion_service": ingestion})

    assert result.outcome == "partial"
    assert any("ingestion failed" in e for e in result.errors)


def test_a_normal_run_reports_per_source_counts_too(db):
    """The same diagnostic is worth having on a real run."""
    ingestion = _ingestion_service(
        db, sources={"cabot": [_candidate(f"k{i}", title=f"Show {i}") for i in range(88)]}
    )

    result, _, _, _ = _run(db, deps={"ingestion_service": ingestion})

    assert result.per_source == {"cabot": SourceTally(fetched=88, accepted=88)}


# ----------------------------------------------------------------------
# Dry run
# ----------------------------------------------------------------------


def test_dry_run_writes_no_events(db):
    _run(db, dry_run=True)

    assert load_events(db) == []


def test_dry_run_writes_no_recommendations(db):
    _, _, _, recs_spy = _run(db, dry_run=True)

    assert recs_spy.calls == 0


def test_dry_run_still_runs_every_stage(db):
    """It exists to surface provider problems, so the work must actually happen."""
    result, fakes, _, _ = _run(db, dry_run=True)

    assert len(fakes["extraction_model"].calls) == 1
    assert fakes["ranking_engine"].titles == [["Karaoke Night"]]
    assert result.outcome == "success"


def test_dry_run_tells_ingestion_not_to_persist(db):
    _, fakes, _, _ = _run(db, dry_run=True)

    assert fakes["ingestion_service"].persist_flags == [False]


def test_a_normal_run_lets_ingestion_persist(db):
    _, fakes, _, _ = _run(db)

    assert fakes["ingestion_service"].persist_flags == [True]


def test_dry_run_pipelines_the_candidates_it_just_fetched(db):
    """Nothing was written, so the loader cannot see them — pass them through."""
    fetched = _candidate("c2", title="Quiz")

    _, fakes, _, _ = _run(
        db,
        dry_run=True,
        deps={"ingestion_service": _ingestion_service(db, [fetched])},
    )

    seen = [c.id for c in fakes["normalization_service"].calls[0]]
    assert seen == ["c1", "c2"]


def test_a_normal_run_does_not_double_count_fetched_candidates(db):
    """Ingestion persisted them, so the loader already returns them."""
    fetched = _candidate("c1")

    _, fakes, _, _ = _run(db, deps={"ingestion_service": _ingestion_service(db, [fetched])})

    seen = [c.id for c in fakes["normalization_service"].calls[0]]
    assert seen == ["c1"]


def test_dry_run_deletes_no_superseded_events(db):
    save_events([_event("stored-a", ["c1"]), _event("stored-b", ["c2"])], db)

    _run(db, dry_run=True, candidates=[_candidate("c1"), _candidate("c2")])

    assert sorted(e.event_id for e in load_events(db)) == ["stored-a", "stored-b"]


# ----------------------------------------------------------------------
# Carry-forward scope
# ----------------------------------------------------------------------


def test_a_stored_event_that_has_already_happened_never_enters_the_pipeline(db):
    """It would ride enrichment, dedup and similarity only to be dropped before ranking."""
    save_events([_event("past", ["c99"], start_time=NOW - timedelta(days=2))], db)

    _, fakes, _, _ = _run(db)

    assert fakes["enrichment_service"].titles[0] == ["Karaoke Night"]
    assert fakes["ranking_engine"].titles == [["Karaoke Night"]]


def test_a_stored_event_that_has_already_happened_is_not_deleted(db):
    """Scoping the carry-forward is not a purge; #17 owns retention."""
    save_events([_event("past", ["c99"], start_time=NOW - timedelta(days=2))], db)

    _run(db)

    assert "past" in {e.event_id for e in load_events(db)}


def test_an_edited_listing_keeps_its_history_and_still_makes_one_event(db):
    """The two halves of #27 that only meet in a whole run.

    Retaining what a source published must not fork the pipeline: the candidate
    id is unchanged by an edit, so reconcile still matches the listing onto the
    event it already produced, and the event count does not move. History
    accumulates beside the row, not in place of it.
    """
    first = _candidate("c1", title="Trivia", start_time=NOW + timedelta(days=2))
    # Seen a day later, which is when an edit is actually noticed — and what
    # orders the history. Two contents observed at the same instant would be a
    # tie the store has no honest way to break.
    edited = _candidate(
        "c1",
        title="Trivia w/ Lee Wolf",
        start_time=NOW + timedelta(days=2),
        discovered_at=NOW + timedelta(days=1),
    )

    repo = _seeded_candidates([first])
    repo.save([edited])

    result, _, _, _ = _run(db, candidates=[edited], deps={}, candidate_repository=repo)

    assert result.stage_counts["events"] == 1
    assert [v.payload["title"] for v in repo.versions_for("c1")] == [
        "Trivia",
        "Trivia w/ Lee Wolf",
    ]


def test_an_event_whose_last_candidate_aged_out_keeps_its_row(db):
    """The persistence consequence of #26, and the reason it was not folded into
    the extraction-scope fix.

    Narrowing the window changes *what reconcile sees*: a stored event whose only
    candidate stops being reloaded is claimed by no fresh event, so it falls to
    `_carry_forward`, which keeps it only if it is still rankable. This one is
    not — so it leaves the working set. It must not leave the database with it.
    `replace` is given an empty stale list and `write_events` is INSERT OR
    REPLACE, so a row simply absent from the batch is left untouched on disk.
    """
    aged_out = _candidate("c1", title="Long over", start_time=NOW - timedelta(days=9))
    save_events([_event("e1", ["c1"], start_time=NOW - timedelta(days=9))], db)

    result, _, _, _ = _run(db, candidates=[], stored_candidates=[aged_out])

    assert result.stage_counts["candidates"] == 0
    assert "e1" in {e.event_id for e in load_events(db)}


def test_an_event_whose_last_candidate_aged_out_leaves_the_working_set(db):
    """The saving the whole change exists for: 649 candidates on 2026-08-14, each
    costing normalization, dedup, enrichment and embedding every night."""
    aged_out = _candidate("c1", title="Long over", start_time=NOW - timedelta(days=9))
    save_events([_event("e1", ["c1"], start_time=NOW - timedelta(days=9))], db)

    result, fakes, _, _ = _run(db, candidates=[], stored_candidates=[aged_out])

    assert result.stage_counts["events"] == 0
    assert fakes["enrichment_service"].titles[0] == []


def test_an_upcoming_event_whose_candidate_is_reloaded_is_still_claimed(db):
    """The other half. Narrowing the window must not stop a live event adopting
    the identity and enrichment its stored row already carries — that would mint
    a new event nightly for the same listing."""
    live = _candidate("c1", title="Still to come", start_time=NOW + timedelta(days=2))
    save_events(
        [_event("e1", ["c1"], title="Still to come", start_time=NOW + timedelta(days=2))],
        db,
    )

    result, _, _, _ = _run(db, candidates=[], stored_candidates=[live])

    assert result.stage_counts["events"] == 1
    assert {e.event_id for e in load_events(db)} == {"e1"}


def test_a_stored_event_beyond_the_horizon_is_not_carried(db):
    """The carry-forward scope matches the ranking scope, so nothing is enriched in vain."""
    save_events([_event("far", ["c99"], title="Far", start_time=NOW + timedelta(days=400))], db)

    _, fakes, _, _ = _run(db)

    assert fakes["enrichment_service"].titles[0] == ["Karaoke Night"]


def test_a_stored_undated_event_older_than_the_lookback_is_not_carried(db):
    """Undated events are held on discovery age, the same rule ranking applies."""
    stale = _event(
        "stale-undated", ["c99"], title="Stale Undated", start_time=None,
        created_at=NOW - timedelta(days=400),
    )
    save_events([stale], db)

    _, fakes, _, _ = _run(db)

    assert fakes["enrichment_service"].titles[0] == ["Karaoke Night"]


def test_a_stored_undated_event_inside_the_lookback_is_still_carried(db):
    """The CLI has a labelled UNDATED section; a recent one must not be dropped."""
    save_events([_event("undated", ["c99"], title="Undated", start_time=None)], db)

    _, fakes, _, _ = _run(db)

    assert sorted(fakes["enrichment_service"].titles[0]) == ["Karaoke Night", "Undated"]


def test_a_fresh_event_with_no_start_time_is_never_scoped_out_early(db):
    """Extraction has not run yet, so a fresh event's start_time is not knowable here."""
    _, fakes, _, _ = _run(
        db,
        candidates=[_candidate("c1", title="Fresh Undated", start_time=None)],
    )

    assert fakes["enrichment_service"].titles[0] == ["Fresh Undated"]


# ----------------------------------------------------------------------
# Run history
# ----------------------------------------------------------------------


def _runs(db) -> list[dict]:
    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    try:
        return [dict(r) for r in conn.execute("SELECT * FROM run_history")]
    finally:
        conn.close()


def test_a_run_is_recorded_in_history(db):
    _run(db)

    assert len(_runs(db)) == 1


def test_the_recorded_outcome_matches_the_result(db):
    _run(db, deps={"enrichment_service": _enrichment_service(db, error=RuntimeError("boom"))})

    assert _runs(db)[0]["outcome"] == "partial"


def test_stage_counts_and_errors_reach_the_history(db):
    _run(db, deps={"enrichment_service": _enrichment_service(db, error=RuntimeError("boom"))})

    row = _runs(db)[0]
    assert "enrichment failed: boom" in json.loads(row["errors"])
    assert json.loads(row["steps_completed"])["ranked"] == 1


def test_skipped_sources_reach_the_history(db):
    _run(db, skipped_sources=["apify"])

    assert json.loads(_runs(db)[0]["skipped_sources"]) == ["apify"]


def test_a_run_that_stops_before_ranking_is_still_recorded(db):
    """The early return is exactly the run whose record matters most."""
    _run(db, deps={"embedding_stage": _embedding_stage(error=RuntimeError("ollama down"))})

    row = _runs(db)[0]
    assert row["outcome"] == "failed"
    assert row["completed_at"] is not None


def test_a_dry_run_records_no_history(db):
    """A dry run did not do a batch; recording one would pollute the only record."""
    _run(db, dry_run=True)

    assert _runs(db) == []


def test_a_hard_crash_leaves_the_run_unfinished(db):
    """A row with a started_at and no completed_at is what a crash looks like.

    `load_events` is deliberately not wrapped as a stage: without stored events
    reconcile treats everything as new, so failing loudly beats duplicating the
    whole database quietly.
    """
    class _UnreadableRepository:
        """A repository whose read fails, standing in for a locked database."""

        def load_all(self):
            raise sqlite3.OperationalError("database is locked")

        def save(self, events): ...

        def save_one(self, event): ...

        def replace(self, stale_ids, events): ...

        def delete(self, event_ids): ...

        def tag_embeddings(self):
            return {}

    with pytest.raises(sqlite3.OperationalError):
        run_batch(
            config=_config(),
            db_path=db,
            logger=get_logger("batch_test", stream=io.StringIO()),
            get_now=lambda: NOW,
            run_date=RUN_DATE,
            ingestion_service=_ingestion_service(db, [_candidate('c1')]),
            normalization_service=_normalization_service(),
            enrichment_service=_enrichment_service(db),
            extraction_stage=_extraction_stage(),
            dedup_decision_repository=SqliteDedupDecisionRepository(db),
            curve_state_repository=SqliteCurveStateRepository(db),
            extraction_observation_repository=SqliteExtractionObservationRepository(db),
            embedding_stage=_embedding_stage(),
            semantic_deduplicator=_DedupSpy(SemanticDeduplicationEngine()),
            similarity_stage=_similarity_stage(),
            ranking_engine=_ranking_engine(),
            event_repository=_UnreadableRepository(),
            candidate_repository=InMemoryCandidateRepository(),
            run_repository=SqliteRunRepository(db),
            score_repository=InMemoryScoreRepository(),
            ranking_repository=InMemoryRankingRepository(),
        )

    row = _runs(db)[0]
    assert row["started_at"] is not None
    assert row["completed_at"] is None
    assert row["outcome"] is None


# ----------------------------------------------------------------------
# Extraction checkpointing
# ----------------------------------------------------------------------


def test_extraction_is_given_a_way_to_save_as_it_goes(db):
    """Extraction runs for hours; a run that dies must not lose all of it."""
    extraction = _extraction_stage()

    _run(db, deps={"extraction_stage": extraction})

    assert extraction.save_fn is not None


def test_a_dry_run_gives_extraction_no_saver(db):
    """A dry run promises to persist nothing, checkpoints included."""
    extraction = _extraction_stage()

    _run(db, dry_run=True, deps={"extraction_stage": extraction})

    assert extraction.save_fn is None


def test_an_extraction_checkpoint_persists_the_single_event_it_is_given(db):
    """The saver handed to extraction takes one event, not a list.

    Asserting only that a counter moved let the two sides disagree about the
    argument entirely: the stage passed an Event, the scheduler unpacked a list,
    and every test still passed while a real batch would have failed on its
    first checkpoint.
    """
    extraction = _extraction_stage()

    _, _, save_spy, _ = _run(db, deps={"extraction_stage": extraction})
    before = save_spy.calls
    extraction.save_fn(_event("checkpointed", ["c1"]))

    assert save_spy.calls == before + 1
    assert "checkpointed" in [e.event_id for e in load_events(db)]


# ----------------------------------------------------------------------
# Superseded events
# ----------------------------------------------------------------------


def test_superseded_events_survive_a_batch_that_dies_before_persisting(db):
    """The delete of a superseded event rides with the save that replaces it.

    Deleting at reconcile time and saving hours later leaves a window covering
    the whole of enrichment and extraction, and a batch that dies inside it has
    dropped the duplicate without writing the merged winner. Holding one
    transaction across those hours is not the alternative — it would lock the
    database for the entire run, which is the failure this refactor exists to
    remove. So the delete moves to the save instead.
    """
    winner = _event("winner", ["c1"])
    loser = _event("loser", ["c1"])
    save_events([winner, loser], db)

    def explode(events):
        raise RuntimeError("disk full")

    with pytest.raises(RuntimeError, match="disk full"):
        _run(db, on_save=explode)

    assert {e.event_id for e in load_events(db)} == {"winner", "loser"}


class TestAFailedReplaceDoesNotCostTheNight:
    """The post-extraction write is the one persistence call that can end a run.

    On 2026-08-12 it did: one event with stale tag vectors made
    `events_repo.replace` raise, and the batch died at that line having spent
    four and a half hours on extraction — never reaching embedding, scoring or
    ranking. Every other stage is wrapped so that "whatever has been saved stays
    saved, and the batch carries on with what it has". This one was not.

    The recovery is real rather than nominal: embedding rebuilds the vectors
    that made the event unwritable, so the final save has a repaired event to
    store.
    """

    def _explode(self, events):
        raise ValueError("event evt-x has 1 tags but 5 tag vectors")

    def test_the_batch_survives_it(self, db):
        result, _, _, _ = _run(db, on_replace=self._explode)

        assert result.outcome == "partial"

    def test_the_failure_is_reported_rather_than_swallowed(self, db):
        result, _, _, _ = _run(db, on_replace=self._explode)

        assert any("has 1 tags but 5 tag vectors" in e for e in result.errors)

    def test_the_run_still_reaches_ranking(self, db):
        """The stages after the failure are the ones that produce the listing."""
        _, _, _, recs_spy = _run(db, on_replace=self._explode)

        assert recs_spy.calls == 1

    def test_the_events_are_still_persisted_by_the_final_save(self, db):
        """The extraction is what cost hours, so it must survive the failure."""
        _, _, save_spy, _ = _run(db, on_replace=self._explode)

        assert SqliteEventRepository(db).load_all() != []


def test_a_stored_event_re_extracts_without_stranding_its_vectors(db):
    """The 2026-08-12 crash, at the level that should have caught it.

    A stored event arrives carrying tags, a vector for each, and the hash of the
    text they came from. The text has since changed, so extraction runs again and
    returns a different number of tags — which is exactly what `min_tags` 5→1
    made routine, and what left five vectors describing one tag.

    This scenario was unwritable while the stage was faked: the fake skipped any
    event that had tags, so a stored event never reached extraction at all, and
    nothing in this module ever exercised the path a nightly batch spends all its
    time on.
    """
    stored = _event(
        "stored-1",
        ["c1"],
        tags=[Tag(text=t, weight=1.0) for t in ("karaoke", "trivia", "bar", "pub", "quiz")],
        summary="An evening of karaoke and trivia",
        setting="indoor",
    )
    stored.attach_tag_embeddings([b"\x00\x01"] * 5)
    stored.extraction_input_hash = "the text has changed since this was extracted"
    save_events([stored], db)

    result, fakes, _, _ = _run(db)

    assert len(fakes["extraction_model"].calls) == 1
    # Not "partial": a re-extraction is the ordinary case, not a degraded one.
    assert result.outcome == "success"
    assert result.errors == []

    reloaded = load_events(db)[0]
    assert [t.text for t in reloaded.tags] == ["karaoke"]
    assert len(reloaded.tag_embeddings) == len(reloaded.tags)


# ----------------------------------------------------------------------
# Behaviour that only became testable once the stages were real
# ----------------------------------------------------------------------


def test_a_dry_run_leaves_no_candidates_behind(db):
    """`persist=False` honoured, not merely passed.

    The old fake recorded the flag and ignored it, so the promise a dry run
    makes — that it writes nothing — was asserted at the call site rather than
    at the database.
    """
    _run(db, dry_run=True, candidates=[_candidate("c9", title="Ghost")])

    conn = sqlite3.connect(db)
    try:
        stored = conn.execute("SELECT count(*) FROM event_candidates").fetchone()[0]
    finally:
        conn.close()
    assert stored == 0


def test_two_candidates_for_one_event_become_one_event(db):
    """Dedup pass 1, over real normalization.

    Two sources listing the same night is the ordinary case, and it is what
    stops the events table growing by a duplicate every run.
    """
    same = dict(title="Karaoke Night", start_time=NOW + timedelta(days=1), venue="The Bar")
    _, fakes, _, _ = _run(
        db, candidates=[_candidate("c1", **same), _candidate("c2", **same)]
    )

    assert fakes["ranking_engine"].titles == [["Karaoke Night"]]
    assert len(load_events(db)) == 1


def test_a_second_run_embeds_nothing(db):
    """Vectors are paid for once, on the hash of the tags and summary.

    The extraction skip has always been asserted; its embedding counterpart
    never was, because the stage that owned the rule was a fake.
    """
    _run(db)
    _, fakes, _, _ = _run(db)

    assert fakes["embedding_model"].calls == []


def test_a_candidate_outside_the_window_never_becomes_an_event(db):
    """Ingestion's window check, which is a different filter from the
    scheduler's ranking scope even though they agree on the horizon."""
    far = _candidate("c1", title="Far", start_time=NOW + timedelta(days=400))

    result, fakes, _, _ = _run(db, candidates=[far], stored_candidates=[])

    assert fakes["normalization_service"].calls[0] == []
    assert fakes["ranking_engine"].titles == [[]]
    assert result.stage_counts["ingested"] == 0


def test_one_failing_source_does_not_lose_the_others(db):
    """A dead site must cost its own listings and nobody else's."""
    result, fakes, _, _ = _run(
        db,
        candidates=[_candidate("c1", title="Quiz Night")],
        deps={
            "ingestion_service": _ingestion_service(
                db,
                sources={
                    "broken": RuntimeError("site down"),
                    "working": [_candidate("c1", title="Quiz Night")],
                },
            )
        },
    )

    assert result.failed_sources == ["broken"]
    assert result.outcome == "success"
    assert fakes["ranking_engine"].titles == [["Quiz Night"]]


def test_the_run_reports_events_the_budget_deferred(db):
    """A deferral is not a failure — the run continues and ranks what it has —
    but a count that stays high run after run is the signal the budget is set
    too low, and nothing else in the summary would say so."""
    log = _stage_log()
    stage = ExtractionStage(
        _ExtractionModel(), None, log, get_now=lambda: NOW, budget_minutes=0
    )
    result, _, _, _ = _run(db, deps={"extraction_stage": _StageSpy(stage)})

    assert result.outcome == "success"
    assert result.stage_counts["extraction_deferred"] == stage.deferred
    assert stage.deferred > 0


# ----------------------------------------------------------------------
# Extraction scope
# ----------------------------------------------------------------------


def test_the_scope_the_batch_hands_extraction_is_the_ranking_scope(db):
    """The wiring, asserted directly, because #26 removed the live path that
    used to demonstrate it end to end.

    Handing the stage *some* predicate is not the claim — handing it the one
    ranking uses is. A predicate that disagreed with ranking would either buy
    extractions ranking discards or skip events ranking wanted.
    """
    stage = _StageSpy(
        ExtractionStage(_ExtractionModel(), None, get_logger("s", stream=io.StringIO()),
                        get_now=lambda: NOW)
    )

    _run(db, deps={"extraction_stage": stage})

    over = _event("e1", [], title="Over", start_time=NOW - timedelta(days=2))
    soon = _event("e2", [], title="Soon", start_time=NOW + timedelta(hours=3))
    beyond = _event("e3", [], title="Beyond", start_time=NOW + timedelta(days=3650))

    assert stage.scope_fn is not None
    assert stage.scope_fn(over) is False
    assert stage.scope_fn(soon) is True
    assert stage.scope_fn(beyond) is False


def test_the_window_floor_is_the_run_date_not_the_moment_the_batch_runs(db):
    """The batch runs at 02:00, and events that started at 00:30 that same night
    are still what the run is for.

    `_scope_filter` keeps an event whose *local date* is the run date, so a
    window bounded by the batch's own clock would drop candidates ranking still
    wants — a disagreement of up to a day between two filters over one field.
    Sharing `_scope_floor` is what stops that, and this is the case that tells
    the two bounds apart: 02:00 local on the run date is behind `now` and ahead
    of the floor.
    """
    early = _candidate(
        "c1",
        title="Small hours",
        # 06:00 UTC = 02:00 in America/New_York on the run date, six hours
        # before the harness clock and two hours after local midnight.
        start_time=NOW - timedelta(hours=6),
    )

    result, fakes, _, _ = _run(db, candidates=[early])

    assert result.stage_counts["candidates"] == 1
    assert fakes["extraction_model"].calls == ["Small hours"]


def test_an_event_ranking_will_discard_never_reaches_the_model(db):
    """A run already under way: began before tonight, ends after it.

    This is the live path, and picking it is the whole point. Ingestion keeps
    such a candidate deliberately — `_within_event_window` tests `end_time`, so
    "a run still under way is not discarded for having begun before tonight" —
    and it therefore arrives on the *fresh* side, which `_carry_forward` never
    scopes. Ranking then discards it on `start_time`.

    A plainly-past candidate would no longer prove anything here: since #26 the
    window drops it on reload, so the assertion would hold with the scope check
    deleted. Measured 2026-08-14, before either fix: an entire 480-minute budget
    spent on events that were already over.
    """
    over = _candidate(
        "c1",
        title="Under way",
        start_time=NOW - timedelta(days=2),
        end_time=NOW + timedelta(days=1),
    )

    _, fakes, _, _ = _run(db, candidates=[over])

    assert fakes["extraction_model"].calls == []


def test_an_event_still_to_come_reaches_the_model(db):
    """The other half: scoping extraction must not stop it doing its job."""
    _, fakes, _, _ = _run(db, candidates=[_candidate("c1", title="Tonight")])

    assert len(fakes["extraction_model"].calls) == 1


def test_a_fresh_undated_event_still_reaches_the_model(db):
    """The case that makes the ranking predicate the right one to reuse: an
    event whose date is not knowable until extraction runs is undated, and the
    predicate keeps undated events on discovery age. A cruder date test would
    skip exactly the events extraction exists to date."""
    undated = _candidate("c1", title="Undated", start_time=None)

    _, fakes, _, _ = _run(db, candidates=[undated])

    assert len(fakes["extraction_model"].calls) == 1


def test_the_run_reports_what_extraction_skipped_as_out_of_scope(db):
    """Without it the fix is invisible: a run that silently skips 124 events
    looks exactly like one with a smaller backlog. It is also the pair that
    separates "the budget is too small" from "the budget is being spent on the
    past" — the confusion that hid two wasted nights.

    Driven as a **dry run**, which is now the only ordinary way an out-of-scope
    event reaches the stage: `fetched` is carried in memory only when nothing
    was persisted, so it is the one path that bypasses `for_window`. On a
    normal run #26 gets there first and this count is structurally zero — the
    check is defence in depth at the seam that sees every event whichever door
    it came in by, not a live filter. Its behaviour is pinned in
    `tests/unit/processing/test_extraction_stage.py`.
    """
    over = _candidate(
        "c1",
        title="Under way",
        start_time=NOW - timedelta(days=2),
        end_time=NOW + timedelta(days=1),
    )
    soon = _candidate("c2", title="Soon")

    result, _, _, _ = _run(db, candidates=[over, soon], dry_run=True)

    assert result.stage_counts["extraction_out_of_scope"] == 1


# ----------------------------------------------------------------------
# Dedup decisions
# ----------------------------------------------------------------------


def _stored_decisions(db):
    return SqliteDedupDecisionRepository(db).load_all()


def test_both_passes_record_their_decisions(db):
    """Pass 1 keys on candidates, Pass 2 on events. Both must reach the table,
    or half the corpus is missing and nothing says so."""
    _run(db, candidates=[_candidate("c1", title="Karaoke Night"),
                         _candidate("c2", title="Poetry Slam")])

    kinds = {d.pass_name for d in _stored_decisions(db)}

    assert kinds == {"fuzzy", "semantic"}


def test_a_rejection_is_recorded_not_only_a_merge(db):
    """The label a surviving row can never carry."""
    _run(db, candidates=[_candidate("c1", title="Karaoke Night"),
                         _candidate("c2", title="Poetry Slam")])

    verdicts = {d.verdict for d in _stored_decisions(db)}

    assert "distinct" in verdicts


def test_a_decision_names_the_run_that_made_it(db):
    """So the thresholds behind the verdict stay recoverable when they change."""
    _run(db, candidates=[_candidate("c1"), _candidate("c2", title="Karaoke Nite")])

    stored = _stored_decisions(db)

    assert stored, "no decisions stored, so this asserted nothing"
    assert all(d.run_id for d in stored)


def test_the_run_records_the_dedup_config_it_used(db):
    """`scoring_config`'s argument, applied to the other set of constants that
    decides what a stored row means."""
    _run(db, candidates=[_candidate("c1")])

    conn = connect(db)
    try:
        stored = conn.execute(
            "SELECT dedup_config FROM run_history ORDER BY started_at DESC"
        ).fetchone()
    finally:
        conn.close()

    assert stored is not None and stored[0], "the run recorded no dedup config"
    assert "semantic_threshold" in stored[0]


def test_a_dry_run_records_no_decisions(db):
    """A dry run persists nothing, and it has no run row to reference."""
    _run(db, candidates=[_candidate("c1"), _candidate("c2", title="Karaoke Nite")],
         dry_run=True)

    assert _stored_decisions(db) == []


def test_an_event_a_merge_displaced_is_kept_not_deleted(db):
    """Reconcile's delete was the last destructive path in the pipeline. A
    cluster is a labelled training scenario, and a deleted loser cannot be
    one — so a displaced event is marked and stays."""
    rich = _event("rich", ["c1"], title="Karaoke Night", tags=[Tag(text="karaoke")])
    thin = _event("thin", ["c1"], title="Karaoke Night")
    save_events([rich, thin], db)

    _run(db, candidates=[_candidate("c1", title="Karaoke Night")])

    ids = {e.event_id for e in load_events(db)}
    assert "thin" in ids, "the displaced event was destroyed"


def test_a_displaced_event_records_what_absorbed_it(db):
    rich = _event("rich", ["c1"], title="Karaoke Night", tags=[Tag(text="karaoke")])
    thin = _event("thin", ["c1"], title="Karaoke Night")
    save_events([rich, thin], db)

    _run(db, candidates=[_candidate("c1", title="Karaoke Night")])

    displaced = next(e for e in load_events(db) if e.event_id == "thin")
    assert displaced.superseded_by == "rich"
    assert displaced.merged_by == "reconcile"


def test_a_displaced_event_never_rejoins_the_ranking(db):
    """The whole risk of keeping it. It was merged away for a reason."""
    rich = _event("rich", ["c1"], title="Karaoke Night", tags=[Tag(text="karaoke")])
    thin = _event("thin", ["c1"], title="Karaoke Night")
    save_events([rich, thin], db)

    _run(db, candidates=[_candidate("c1", title="Karaoke Night")])
    _, fakes, _, _ = _run(db, candidates=[_candidate("c1", title="Karaoke Night")])

    assert "thin" not in fakes["ranking_engine"].ranked[0]
