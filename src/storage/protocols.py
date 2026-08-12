"""Storage contracts the core depends on.

Core code — stages, services, the scheduler — depends on these Protocols and
never on a concrete store. That is what lets a stage be tested without SQLite,
and what keeps connection handling and row mapping in one place instead of
spread across the modules that happen to need data.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Protocol

from src.models.event import Event
from src.models.event_candidate import EventCandidate
from src.models.event_score import EventScore
from src.models.ranking import Ranking
from src.models.run import RunRecord


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

    def start(self, started_at: datetime) -> str:
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
