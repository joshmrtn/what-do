"""Storage contracts the core depends on.

Core code — stages, services, the scheduler — depends on these Protocols and
never on a concrete store. That is what lets a stage be tested without SQLite,
and what keeps connection handling and row mapping in one place instead of
spread across the modules that happen to need data.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any, Protocol

from src.models.candidate_entity import CandidateEntity
from src.models.event import Event
from src.models.event_candidate import EventCandidate
from src.models.event_score import EventScore
from src.models.ranking import Ranking
from src.models.run import RunRecord
from src.normalization.decision_sampling import SampledDecision
from src.storage.dedup_decisions import StoredDecision
from src.storage.http_cache import CachedResponse


class EventRepository(Protocol):
    """Persistence for canonical events, with their tags, vectors and provenance."""

    def save(self, events: list[Event]) -> None:
        """Insert or replace events, replacing each one's tags and provenance.

        Args:
            events: Events to persist. An empty list is a no-op — "nothing to
                save" is never "clear the store".

        Raises:
            ValueError: If an event carries a number of tag vectors that does
                not match its number of tags. Storing the overlap would drop
                the rest and report success.
        """
        ...

    def save_one(self, event: Event) -> None:
        """Persist a single event.

        Exists because extraction costs minutes per event and the batch has to
        checkpoint as it goes. Passing the whole corpus to `save` for each one
        rewrites every row to store one, which is what forced batching before.
        """
        ...

    def load_all(self) -> list[Event]:
        """Every stored event, with tags, vectors and provenance reattached."""
        ...

    def delete(self, event_ids: list[str]) -> None:
        """Remove events superseded by a merge.

        Args:
            event_ids: Events to delete. An empty list is a no-op.
        """
        ...

    def replace(self, stale_ids: list[str], events: list[Event]) -> None:
        """Delete superseded events and save their replacements, atomically.

        Reconcile identifies superseded duplicates hours before the run has
        anything to write in their place. Deleting them at that point opens a
        window across enrichment and extraction where the duplicate is gone and
        the merged winner was never stored; holding one transaction across those
        hours instead would lock the database for the whole batch, which is the
        failure this boundary exists to remove. So the delete travels with the
        save and both take milliseconds.

        Args:
            stale_ids: Events superseded by a merge.
            events: Events to persist in their place.
        """
        ...

    def tag_embeddings(self) -> dict[str, bytes]:
        """Every tag vector already computed, keyed by tag text.

        A vector is a pure function of its text and the embedding model, so a
        tag embedded on a previous night never needs embedding again.
        """
        ...


class RunRepository(Protocol):
    """Persistence for `run_history`, the only durable record of a batch run."""

    def start(
        self,
        started_at: datetime,
        scoring_config: str | None = None,
        dedup_config: str | None = None,
    ) -> str:
        """Record that a batch has begun, returning its run id.

        Written at the start rather than the end so a run killed mid-flight
        still leaves evidence that it began.
        """
        ...

    def finish(
        self,
        run_id: str,
        *,
        outcome: str,
        completed_at: datetime,
        stage_counts: dict[str, int] | None = None,
        errors: list[str] | None = None,
        skipped_sources: list[str] | None = None,
    ) -> None:
        """Complete a run's row with its outcome, counts, errors and skips.

        An unknown `run_id` updates nothing rather than raising: the batch must
        never die trying to record that it died.

        Args:
            run_id: The id returned by `start`.
            outcome: One of `success`, `partial`, `failed`.
            completed_at: When the run ended. The duration is derived against
                the stored `started_at`, not a passed-in value, so a resumed
                process still records real elapsed time.
            stage_counts: Per-stage counts.
            errors: Stage failure messages.
            skipped_sources: Sources not built, normally a missing credential.
        """
        ...

    def open_run(self) -> RunRecord | None:
        """The most recent run that began and never finished, if any.

        A started row with no `completed_at` is a crash, and this is the only
        way the system can learn one happened. Candidates are the *unfinished*
        rows, not the latest row overall — a run that succeeded after a crash
        must not report that all is well while the crash sits unexamined.
        """
        ...

    def get(self, run_id: str) -> RunRecord | None:
        """One run's record, or None if no such run exists."""
        ...


class ScoreRepository(Protocol):
    """Persistence for `event_scores` and the reasons behind them.

    Scores and reasons are one aggregate: a reason is meaningless without the
    score it explains, they are written together, and `score_reasons` cascades
    from `event_scores`.
    """

    def save(self, scores: list[EventScore]) -> None:
        """Insert scores for one or more run dates, replacing those runs.

        A re-run of the same date supersedes its earlier attempt rather than
        accumulating a second copy, or the CLI would read two conflicting
        orderings. Replacement is scoped to the run dates being written, so
        previous nights are untouched.

        Args:
            scores: The run's scores. Empty is a no-op — "nothing to save" is
                never "clear the table".
        """
        ...

    def for_run(self, run_date: date) -> list[EventScore]:
        """Every score stored for one batch date, with its reasons reattached."""
        ...


class RankingRepository(Protocol):
    """Persistence for `rankings` — where each in-scope event landed."""

    def save(self, rankings: list[Ranking]) -> None:
        """Insert placements for one or more run dates, replacing those runs.

        Rankings reference `event_scores` by `(event_id, run_date)`, so the
        matching scores must already be saved. Empty is a no-op.
        """
        ...

    def for_run(self, run_date: date) -> list[Ranking]:
        """One batch date's placements, in the rank the batch assigned."""
        ...

    def latest_run_date(self) -> date | None:
        """The most recent batch date that produced a ranking, if any.

        Deliberately not "the most recent run": a run that died before ranking
        leaves a `run_history` row and nothing to show, and the CLI wants the
        last night it can actually display.
        """
        ...


class CandidateRepository(Protocol):
    """Persistence for raw ingested candidates, before normalization.

    The batch reads candidates back from here rather than passing them through
    in memory, which is what makes the ingest boundary crash-survivable: a
    re-run after a failure picks up what was already fetched without touching
    the network again.
    """

    def save(self, candidates: list[EventCandidate]) -> None:
        """Insert candidates, replacing any stored under the same id.

        Args:
            candidates: Candidates to store. Empty is a no-op — "nothing to
                save" is never "clear the store".
        """
        ...

    def for_window(
        self, *, discovered_since: datetime, starting_after: datetime
    ) -> list[EventCandidate]:
        """Candidates still in scope for a run.

        The window is a **union**, because either filter alone starves a source
        type: social candidates carry no `start_time` at ingestion, so a
        forward-only filter drops all of them, while a `discovered_at`-only
        filter eventually drops calendar events that are still upcoming.

        Args:
            discovered_since: Earliest `discovered_at` to accept, normally the
                lookback cutoff.
            starting_after: Earliest `start_time` to accept regardless of age,
                normally the run's now.

        Returns:
            Matching candidates, ordered by discovery then id. The order is
            fixed because dedup picks a merge base partly on the order it sees.
        """
        ...


class EntityRepository(Protocol):
    """Persistence for `candidate_entities` — handles discovered from post text.

    Discovery works by mention: a handle named by enough sources we already
    trust earns its way to `active` and becomes something we fetch from. The
    counters are the evidence for that promotion, which is why the write below
    accumulates rather than replaces.
    """

    def active_handles(self) -> list[str]:
        """Every handle currently active for ingestion, alphabetically.

        Empty when the database has no schema yet — a first run, before any
        batch has initialised it.
        """
        ...

    def mark_seeds_active(self, handles: list[str], *, now: datetime) -> None:
        """Upsert seed handles as active at depth 0.

        A handle already discovered by mention is promoted **in place**, keeping
        its counters, so adding it to `seeds.yaml` activates the row rather than
        colliding with it.
        """
        ...

    def record_mention(
        self,
        *,
        handle: str,
        source_handle: str,
        depth: int,
        context: str | None,
        now: datetime,
    ) -> None:
        """Record that `source_handle` mentioned `handle`.

        Accumulates: a handle seen before gains a mention rather than being
        replaced. A source that has already mentioned this handle counts once
        and no more — otherwise one chatty account promotes its own friends.
        The first non-empty `context` is kept, because the earliest sighting is
        the one nearest the discovery that justified keeping the handle.
        """
        ...

    def by_handle(self, handle: str) -> CandidateEntity | None:
        """One entity by its handle, or None if it has never been seen."""
        ...

    def unclassified(self) -> list[CandidateEntity]:
        """Probationary handles disambiguation has not yet judged."""
        ...

    def classify(
        self, entity_id: str, *, classification: str, state: str, now: datetime
    ) -> None:
        """Record what disambiguation decided, and the state that follows."""
        ...

    def awaiting_promotion(self) -> list[CandidateEntity]:
        """Probationary handles classified as venues, with their evidence.

        The caller decides whether the evidence clears the threshold; this only
        says which handles are eligible to be judged.
        """
        ...

    def activate(self, entity_id: str, *, now: datetime) -> None:
        """Promote a handle to `active`, making it a source for the next run."""
        ...


class WeatherCache(Protocol):
    """Cached forecasts, keyed by day and place.

    `get` takes the oldest stamp still worth serving rather than handing back
    whatever is stored, so a caller cannot serve a stale forecast by forgetting
    to check its age — see `weather.cache_ttl_hours`, which must stay under 24
    hours so a nightly batch refetches.
    """

    def get(
        self,
        *,
        day: date,
        latitude: float,
        longitude: float,
        fresh_since: datetime,
    ) -> dict[str, Any] | None:
        """The forecast for a day and place, or None if absent or too old."""
        ...

    def put(
        self,
        *,
        day: date,
        latitude: float,
        longitude: float,
        data: dict[str, Any],
        now: datetime,
    ) -> None:
        """Store a forecast, replacing any entry for the same day and place."""
        ...


class HttpCache(Protocol):
    """Stored responses with whatever validators a server offered.

    Exists so politeness survives a process restart: without persisted
    validators, a person re-running the batch a few times in an evening turns
    every run into a full download of somebody else's server.
    """

    def get(self, url: str) -> CachedResponse | None:
        """The cached response for a URL, or None if it was never fetched."""
        ...

    def put(
        self,
        url: str,
        *,
        body: str,
        etag: str | None,
        last_modified: str | None,
        fetched_at: datetime,
    ) -> None:
        """Store a response, replacing any earlier entry for the same URL."""
        ...


class DedupDecisionRepository(Protocol):
    """Persistence for `dedup_decisions` — what the dedup passes concluded.

    Kept apart from the events themselves because a decision is about a *pair*,
    and a surviving row can only ever record the merges it won. The rejections
    live here, and they are most of what a future dedup model learns from.
    """

    def save(
        self, decisions: list[SampledDecision], *, run_id: str, now: datetime
    ) -> None:
        """Store decisions, replacing any previous verdict on the same pair."""
        ...

    def load_all(self) -> list[StoredDecision]:
        """Every stored decision, for inspection and for training."""
        ...
