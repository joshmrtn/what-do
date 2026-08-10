"""Unit tests for the batch orchestrator's sequencing and save points."""

from __future__ import annotations

import copy
import io
import json
import sqlite3
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pytest

from src.config import (
    AppConfig,
    DeduplicationConfig,
    LocationConfig,
    ScrapingConfig,
    VenueDiscoveryConfig,
)
from src.ingestion.ingestion_service import IngestionResult
from src.models.event import Event
from src.models.event_candidate import EventCandidate
from src.models.recommendation import Recommendation
from src.models.tag import Tag
from src.normalization.service import NormalizationResult
from src.scheduler import run_batch
from src.storage.db import init_db
from src.storage.events import delete_events, load_events, save_events
from src.utils.logging import get_logger

NOW = datetime(2026, 6, 15, 12, 0, 0, tzinfo=timezone.utc)
RUN_DATE = date(2026, 6, 15)


# ----------------------------------------------------------------------
# Fakes
# ----------------------------------------------------------------------


class _FakeIngestion:
    def __init__(
        self,
        error: Exception | None = None,
        candidates: list[EventCandidate] | None = None,
        per_source: dict[str, int] | None = None,
        failed_sources: list[str] | None = None,
    ) -> None:
        self.error = error
        self.candidates = candidates or []
        self.per_source = per_source or {}
        self.failed_sources = failed_sources or []
        self.calls = 0
        self.persist_flags: list[bool] = []
        self.raw_flags: list[bool] = []

    def run(self, get_now=None, persist=True, collect_raw=False):
        self.calls += 1
        self.persist_flags.append(persist)
        self.raw_flags.append(collect_raw)
        if self.error:
            raise self.error
        return IngestionResult(
            accepted=len(self.candidates),
            discarded=0,
            handles_discovered=0,
            candidates=list(self.candidates),
            per_source=dict(self.per_source),
            failed_sources=list(self.failed_sources),
        )


class _FakeNormalization:
    def __init__(self, events: list[Event]) -> None:
        self._events = events
        self.calls: list[list[EventCandidate]] = []

    def run(self, candidates, get_now=None):
        self.calls.append(list(candidates))
        events = copy.deepcopy(self._events)
        return NormalizationResult(normalized=len(events), discarded=0, events=events)


class _FakeEnrichment:
    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.seen: list[list[str]] = []

    def enrich(self, events, run_date):
        if self.error:
            raise self.error
        self.seen.append([e.event_id for e in events])
        for event in events:
            event.astronomical_data = {"sunset": "20:15"}
        return events


class _FakeExtraction:
    """Mirrors ExtractionStage: skips any event that already carries tags."""

    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.extracted: list[list[str]] = []
        self.save_fn = None

    def set_save_fn(self, save_fn):
        self.save_fn = save_fn

    def process(self, events):
        if self.error:
            raise self.error
        did = []
        for event in events:
            if event.tags:
                continue
            event.tags = [Tag(text="karaoke", weight=1.0)]
            event.summary = "Karaoke night"
            event.setting = "indoor"
            did.append(event.event_id)
        self.extracted.append(did)
        return events


class _FakeEmbedding:
    """Mirrors EmbeddingStage: skips any event that already carries vectors."""

    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.embedded: list[list[str]] = []

    def process(self, events):
        if self.error:
            raise self.error
        did = []
        for event in events:
            if event.tag_embeddings or not event.tags:
                continue
            event.tag_embeddings = [b"\x00\x01"]
            event.summary_embedding = b"\x02\x03"
            did.append(event.event_id)
        self.embedded.append(did)
        return events


class _FakeSemanticDedup:
    def __init__(self) -> None:
        self.calls = 0

    def deduplicate(self, events, config):
        self.calls += 1
        return events


class _FakeSimilarity:
    def process(self, events):
        return events


class _FakeRanking:
    def __init__(self) -> None:
        self.ranked: list[list[str]] = []

    def rank(self, events, run_date):
        self.ranked.append([e.event_id for e in events])
        return [
            Recommendation(
                recommendation_id=f"{run_date}:{e.event_id}",
                event_id=e.event_id,
                run_date=run_date,
                base_score=1.0,
                weather_adjustment=0.0,
                tag_confidence=1.0,
                final_score=1.0,
                match="yes",
                tier="top_picks",
                rank=i + 1,
                reasons=[],
            )
            for i, e in enumerate(events)
        ]


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


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _config(**scraping) -> AppConfig:
    return AppConfig(
        location=LocationConfig(42.52, -70.89, "01970", 10.0, "America/New_York"),
        scraping=ScrapingConfig(**scraping),
        venue_discovery=VenueDiscoveryConfig(),
        deduplication=DeduplicationConfig(),
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


@pytest.fixture
def db(tmp_path) -> Path:
    path = tmp_path / "batch.db"
    init_db(path)
    return path


def _run(db, *, fresh=None, deps=None, **kwargs):
    """Drive run_batch with fakes, returning (result, the fakes it used)."""
    fakes = {
        "ingestion_service": _FakeIngestion(),
        "normalization_service": _FakeNormalization(
            fresh if fresh is not None else [_event("fresh-1", ["c1"])]
        ),
        "enrichment_service": _FakeEnrichment(),
        "extraction_stage": _FakeExtraction(),
        "embedding_stage": _FakeEmbedding(),
        "semantic_deduplicator": _FakeSemanticDedup(),
        "similarity_stage": _FakeSimilarity(),
        "ranking_engine": _FakeRanking(),
    }
    fakes.update(deps or {})

    save_spy = _Spy(save_events)
    recs_spy = _Spy(lambda items, path: None)

    result = run_batch(
        config=_config(),
        db_path=db,
        logger=get_logger("batch_test", stream=io.StringIO()),
        get_now=lambda: NOW,
        run_date=RUN_DATE,
        load_candidates_fn=lambda *a, **k: [
            EventCandidate(id="c1", source="@v", source_type="apify", discovered_at=NOW)
        ],
        load_events_fn=load_events,
        save_events_fn=save_spy,
        delete_events_fn=delete_events,
        save_recommendations_fn=recs_spy,
        **fakes,
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
    assert [r.event_id for r in recs_spy.snapshots[0]] == ["fresh-1"]


# ----------------------------------------------------------------------
# Save points
# ----------------------------------------------------------------------


def test_events_are_saved_after_extraction(db):
    """Three minutes an event of LLM time must survive a crash in a later stage."""
    _, _, save_spy, _ = _run(db)

    first = save_spy.snapshots[0]
    assert first == [("fresh-1", True, False)]


def test_events_are_saved_after_embedding(db):
    _, _, save_spy, _ = _run(db)

    assert ("fresh-1", True, True) in save_spy.snapshots[1]


def test_a_crash_after_extraction_leaves_tags_persisted(db):
    """Everything already saved stays saved, which is what makes a re-run cheap."""
    _run(db, deps={"embedding_stage": _FakeEmbedding(error=RuntimeError("ollama down"))})

    stored = load_events(db)
    assert [t.text for t in stored[0].tags] == ["karaoke"]


def test_a_second_run_re_extracts_nothing(db):
    """The whole point of persisting: model time is paid once per event."""
    _run(db)
    _, fakes, _, _ = _run(db)

    assert fakes["extraction_stage"].extracted == [[]]


def test_a_second_run_reuses_the_stored_event_id(db):
    _run(db)
    _, fakes, _, _ = _run(db)

    assert fakes["ranking_engine"].ranked == [["fresh-1"]]
    assert len(load_events(db)) == 1


# ----------------------------------------------------------------------
# Reconcile wiring
# ----------------------------------------------------------------------


def test_stored_enrichment_is_adopted_by_the_matching_fresh_event(db):
    save_events(
        [
            _event(
                "stored-1",
                ["c1"],
                tags=[Tag(text="trivia", weight=1.0)],
                summary="Trivia night",
                setting="indoor",
            )
        ],
        db,
    )

    _, fakes, _, _ = _run(db, fresh=[_event("fresh-1", ["c1"])])

    assert fakes["ranking_engine"].ranked == [["stored-1"]]
    assert fakes["extraction_stage"].extracted == [[]]


def test_a_superseded_event_is_deleted(db):
    """Without the delete the loser lingers forever as a duplicate in the output."""
    save_events([_event("stored-a", ["c1"]), _event("stored-b", ["c2"])], db)

    _run(db, fresh=[_event("fresh-1", ["c1", "c2"])])

    assert sorted(e.event_id for e in load_events(db)) == ["stored-a"]


def test_a_stored_event_with_no_fresh_counterpart_is_still_ranked(db):
    """Its candidates aged out of the window; the event itself has not."""
    save_events([_event("stored-old", ["c99"])], db)

    _, fakes, _, _ = _run(db, fresh=[_event("fresh-1", ["c1"])])

    assert sorted(fakes["ranking_engine"].ranked[0]) == ["fresh-1", "stored-old"]


# ----------------------------------------------------------------------
# Failure policy
# ----------------------------------------------------------------------


def test_ingestion_failure_does_not_abort_the_run(db):
    """A night where every source is down should still answer from what we have."""
    save_events([_event("stored-1", ["c1"])], db)

    result, fakes, _, recs_spy = _run(
        db,
        fresh=[],
        deps={"ingestion_service": _FakeIngestion(error=RuntimeError("all sources down"))},
    )

    assert result.outcome == "partial"
    assert fakes["ranking_engine"].ranked == [["stored-1"]]
    assert recs_spy.calls == 1


def test_a_stage_failure_is_recorded_in_the_result(db):
    result, _, _, _ = _run(
        db, deps={"enrichment_service": _FakeEnrichment(error=RuntimeError("weather down"))}
    )

    assert result.outcome == "partial"
    assert any("weather down" in e for e in result.errors)


def test_enrichment_failure_still_reaches_ranking(db):
    _, fakes, _, _ = _run(
        db, deps={"enrichment_service": _FakeEnrichment(error=RuntimeError("weather down"))}
    )

    assert fakes["ranking_engine"].ranked == [["fresh-1"]]


def test_wholesale_embedding_failure_stops_before_ranking(db):
    """Ranking without vectors produces garbage, and it would be persisted."""
    result, fakes, _, recs_spy = _run(
        db, deps={"embedding_stage": _FakeEmbedding(error=RuntimeError("ollama down"))}
    )

    assert fakes["ranking_engine"].ranked == []
    assert recs_spy.calls == 0
    assert result.outcome == "failed"


# ----------------------------------------------------------------------
# Ranking scope
# ----------------------------------------------------------------------


def test_events_beyond_the_horizon_are_not_ranked(db):
    far = _event("far", ["c1"], start_time=NOW + timedelta(days=90))

    _, fakes, _, _ = _run(db, fresh=[far])

    assert fakes["ranking_engine"].ranked == [[]]


def test_events_already_past_are_not_ranked(db):
    gone = _event("gone", ["c1"], start_time=NOW - timedelta(days=2))

    _, fakes, _, _ = _run(db, fresh=[gone])

    assert fakes["ranking_engine"].ranked == [[]]


def test_undated_events_inside_the_lookback_are_ranked(db):
    """The CLI has a labelled UNDATED section and must not lose them."""
    undated = _event("undated", ["c1"], start_time=None)

    _, fakes, _, _ = _run(db, fresh=[undated])

    assert fakes["ranking_engine"].ranked == [["undated"]]


def test_undated_events_older_than_the_lookback_are_not_ranked(db):
    stale = _event(
        "stale", ["c1"], start_time=None, created_at=NOW - timedelta(days=200)
    )

    _, fakes, _, _ = _run(db, fresh=[stale])

    assert fakes["ranking_engine"].ranked == [[]]


def test_an_event_tonight_is_ranked(db):
    tonight = _event("tonight", ["c1"], start_time=NOW + timedelta(hours=8))

    _, fakes, _, _ = _run(db, fresh=[tonight])

    assert fakes["ranking_engine"].ranked == [["tonight"]]


# ----------------------------------------------------------------------
# Ingest-only
# ----------------------------------------------------------------------


def test_ingest_only_stops_before_the_pipeline(db):
    """It exists to prove the sources fetch, without paying for extraction."""
    _, fakes, _, _ = _run(db, ingest_only=True)

    assert fakes["ingestion_service"].calls == 1
    assert fakes["normalization_service"].calls == []
    assert fakes["extraction_stage"].extracted == []
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
    ingestion = _FakeIngestion(
        candidates=[
            EventCandidate(id="a", source="s", source_type="apify", discovered_at=NOW),
            EventCandidate(id="b", source="s", source_type="apify", discovered_at=NOW),
        ]
    )

    result, _, _, _ = _run(db, ingest_only=True, deps={"ingestion_service": ingestion})

    assert result.stage_counts["ingested"] == 2


def test_ingest_only_reports_what_each_source_returned(db):
    """A total says nothing about which of seventeen sources went quiet."""
    ingestion = _FakeIngestion(per_source={"do617_gulu_gulu": 25, "do617_koto": 0})

    result, _, _, _ = _run(db, ingest_only=True, deps={"ingestion_service": ingestion})

    assert result.per_source == {"do617_gulu_gulu": 25, "do617_koto": 0}


def test_ingest_only_reports_a_source_that_failed(db):
    ingestion = _FakeIngestion(failed_sources=["nshoremag"])

    result, _, _, _ = _run(db, ingest_only=True, deps={"ingestion_service": ingestion})

    assert result.failed_sources == ["nshoremag"]


def test_ingest_only_records_no_run_history(db):
    """A diagnostic did not do a batch, as with a dry run."""
    _run(db, ingest_only=True)

    assert _runs(db) == []


def test_ingest_only_survives_an_ingestion_failure(db):
    ingestion = _FakeIngestion(error=RuntimeError("network down"))

    result, _, _, _ = _run(db, ingest_only=True, deps={"ingestion_service": ingestion})

    assert result.outcome == "partial"
    assert any("ingestion failed" in e for e in result.errors)


def test_a_normal_run_reports_per_source_counts_too(db):
    """The same diagnostic is worth having on a real run."""
    ingestion = _FakeIngestion(per_source={"cabot": 88})

    result, _, _, _ = _run(db, deps={"ingestion_service": ingestion})

    assert result.per_source == {"cabot": 88}


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

    assert fakes["extraction_stage"].extracted == [["fresh-1"]]
    assert fakes["ranking_engine"].ranked == [["fresh-1"]]
    assert result.outcome == "success"


def test_dry_run_tells_ingestion_not_to_persist(db):
    _, fakes, _, _ = _run(db, dry_run=True)

    assert fakes["ingestion_service"].persist_flags == [False]


def test_a_normal_run_lets_ingestion_persist(db):
    _, fakes, _, _ = _run(db)

    assert fakes["ingestion_service"].persist_flags == [True]


def test_dry_run_pipelines_the_candidates_it_just_fetched(db):
    """Nothing was written, so the loader cannot see them — pass them through."""
    fetched = EventCandidate(
        id="c2", source="@v", source_type="apify", discovered_at=NOW
    )

    _, fakes, _, _ = _run(
        db, dry_run=True, deps={"ingestion_service": _FakeIngestion(candidates=[fetched])}
    )

    seen = [c.id for c in fakes["normalization_service"].calls[0]]
    assert seen == ["c1", "c2"]


def test_a_normal_run_does_not_double_count_fetched_candidates(db):
    """Ingestion persisted them, so the loader already returns them."""
    fetched = EventCandidate(
        id="c1", source="@v", source_type="apify", discovered_at=NOW
    )

    _, fakes, _, _ = _run(
        db, deps={"ingestion_service": _FakeIngestion(candidates=[fetched])}
    )

    seen = [c.id for c in fakes["normalization_service"].calls[0]]
    assert seen == ["c1"]


def test_dry_run_deletes_no_superseded_events(db):
    save_events([_event("stored-a", ["c1"]), _event("stored-b", ["c2"])], db)

    _run(db, dry_run=True, fresh=[_event("fresh-1", ["c1", "c2"])])

    assert sorted(e.event_id for e in load_events(db)) == ["stored-a", "stored-b"]


# ----------------------------------------------------------------------
# Carry-forward scope
# ----------------------------------------------------------------------


def test_a_stored_event_that_has_already_happened_never_enters_the_pipeline(db):
    """It would ride enrichment, dedup and similarity only to be dropped before ranking."""
    save_events([_event("past", ["c99"], start_time=NOW - timedelta(days=2))], db)

    _, fakes, _, _ = _run(db, fresh=[_event("fresh-1", ["c1"])])

    assert fakes["enrichment_service"].seen[0] == ["fresh-1"]
    assert fakes["ranking_engine"].ranked == [["fresh-1"]]


def test_a_stored_event_that_has_already_happened_is_not_deleted(db):
    """Scoping the carry-forward is not a purge; #17 owns retention."""
    save_events([_event("past", ["c99"], start_time=NOW - timedelta(days=2))], db)

    _run(db, fresh=[_event("fresh-1", ["c1"])])

    assert "past" in {e.event_id for e in load_events(db)}


def test_a_stored_event_beyond_the_horizon_is_not_carried(db):
    """The carry-forward scope matches the ranking scope, so nothing is enriched in vain."""
    save_events([_event("far", ["c99"], start_time=NOW + timedelta(days=400))], db)

    _, fakes, _, _ = _run(db, fresh=[_event("fresh-1", ["c1"])])

    assert fakes["enrichment_service"].seen[0] == ["fresh-1"]


def test_a_stored_undated_event_older_than_the_lookback_is_not_carried(db):
    """Undated events are held on discovery age, the same rule ranking applies."""
    stale = _event(
        "stale-undated", ["c99"], start_time=None, created_at=NOW - timedelta(days=400)
    )
    save_events([stale], db)

    _, fakes, _, _ = _run(db, fresh=[_event("fresh-1", ["c1"])])

    assert fakes["enrichment_service"].seen[0] == ["fresh-1"]


def test_a_stored_undated_event_inside_the_lookback_is_still_carried(db):
    """The CLI has a labelled UNDATED section; a recent one must not be dropped."""
    save_events([_event("undated", ["c99"], start_time=None)], db)

    _, fakes, _, _ = _run(db, fresh=[_event("fresh-1", ["c1"])])

    assert sorted(fakes["enrichment_service"].seen[0]) == ["fresh-1", "undated"]


def test_a_fresh_event_with_no_start_time_is_never_scoped_out_early(db):
    """Extraction has not run yet, so a fresh event's start_time is not knowable here."""
    _, fakes, _, _ = _run(
        db,
        fresh=[_event("fresh-undated", ["c1"], start_time=None, created_at=NOW - timedelta(days=400))],
    )

    assert fakes["enrichment_service"].seen[0] == ["fresh-undated"]


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
    _run(db, deps={"enrichment_service": _FakeEnrichment(error=RuntimeError("boom"))})

    assert _runs(db)[0]["outcome"] == "partial"


def test_stage_counts_and_errors_reach_the_history(db):
    _run(db, deps={"enrichment_service": _FakeEnrichment(error=RuntimeError("boom"))})

    row = _runs(db)[0]
    assert "enrichment failed: boom" in json.loads(row["errors"])
    assert json.loads(row["steps_completed"])["ranked"] == 1


def test_skipped_sources_reach_the_history(db):
    _run(db, skipped_sources=["apify"])

    assert json.loads(_runs(db)[0]["skipped_sources"]) == ["apify"]


def test_a_run_that_stops_before_ranking_is_still_recorded(db):
    """The early return is exactly the run whose record matters most."""
    _run(db, deps={"embedding_stage": _FakeEmbedding(error=RuntimeError("ollama down"))})

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
    def _boom(*_args, **_kwargs):
        raise sqlite3.OperationalError("database is locked")

    with pytest.raises(sqlite3.OperationalError):
        run_batch(
            config=_config(),
            db_path=db,
            logger=get_logger("batch_test", stream=io.StringIO()),
            get_now=lambda: NOW,
            run_date=RUN_DATE,
            ingestion_service=_FakeIngestion(),
            normalization_service=_FakeNormalization([]),
            enrichment_service=_FakeEnrichment(),
            extraction_stage=_FakeExtraction(),
            embedding_stage=_FakeEmbedding(),
            semantic_deduplicator=_FakeSemanticDedup(),
            similarity_stage=_FakeSimilarity(),
            ranking_engine=_FakeRanking(),
            load_events_fn=_boom,
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
    extraction = _FakeExtraction()

    _run(db, deps={"extraction_stage": extraction})

    assert extraction.save_fn is not None


def test_a_dry_run_gives_extraction_no_saver(db):
    """A dry run promises to persist nothing, checkpoints included."""
    extraction = _FakeExtraction()

    _run(db, dry_run=True, deps={"extraction_stage": extraction})

    assert extraction.save_fn is None


def test_an_extraction_checkpoint_persists_the_single_event_it_is_given(db):
    """The saver handed to extraction takes one event, not a list.

    Asserting only that a counter moved let the two sides disagree about the
    argument entirely: the stage passed an Event, the scheduler unpacked a list,
    and every test still passed while a real batch would have failed on its
    first checkpoint.
    """
    extraction = _FakeExtraction()

    _, _, save_spy, _ = _run(db, deps={"extraction_stage": extraction})
    before = save_spy.calls
    extraction.save_fn(_event("checkpointed", ["c1"]))

    assert save_spy.calls == before + 1
    assert "checkpointed" in [e.event_id for e in load_events(db)]
