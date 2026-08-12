"""IngestionService — orchestrates the full Phase 3 ingestion run."""

from __future__ import annotations

import json
import sqlite3

from src.storage.candidates import write_candidates
from src.storage.db import connect
import uuid
from dataclasses import dataclass, field as dataclass_field
from datetime import datetime, timedelta, tzinfo, timezone
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
from typing import Any, Callable

from src.config import AppConfig
from src.ingestion.failover import FailoverChain
from src.ingestion.handle_extractor import HandleExtractor
from src.ingestion.seeds import load_seeds
from src.ingestion.source import IngestionSource
from src.models.event_candidate import EventCandidate
from src.utils.nights import night_start


@dataclass(frozen=True)
class SourceTally:
    """What one source contributed.

    Both numbers are needed to tell the two silences apart: a source that
    returned nothing is broken or empty, while a source that returned plenty and
    kept none is parsing dates into the wrong window — which looks identical in
    a total.
    """

    fetched: int = 0
    accepted: int = 0


@dataclass(frozen=True)
class RawCandidateRecord:
    """One candidate exactly as a source produced it, with its verdict.

    Collected before filtering, because filtering is the thing most likely to be
    wrong. A dump of what survived cannot explain an empty run.
    """

    candidate: EventCandidate
    source: str
    verdict: str
    reason: str | None = None


@dataclass
class IngestionResult:
    """Summary of a single ingestion run."""

    #: Candidates that passed filtering. Named for what it counts rather than
    #: what was written, since a `persist=False` run accepts without writing.
    accepted: int
    discarded: int
    handles_discovered: int
    candidates: list[EventCandidate] = dataclass_field(default_factory=list)
    #: What each source fetched and kept, keyed by its `source_name`.
    per_source: dict[str, SourceTally] = dataclass_field(default_factory=dict)
    #: Sources that raised. Named, because a total cannot say which went quiet.
    failed_sources: list[str] = dataclass_field(default_factory=list)
    #: Every candidate as fetched, only when `collect_raw` asked for it.
    raw: list[RawCandidateRecord] = dataclass_field(default_factory=list)


class IngestionService:
    """Runs the full ingestion pipeline: seed load, scraping, filtering, persistence."""

    def __init__(
        self,
        config: AppConfig,
        db_path: Path,
        seeds_path: Path,
        failover_sources: list[IngestionSource],
        independent_sources: list[IngestionSource],
        logger: Any,
        blocklist: list[str] | None = None,
    ) -> None:
        """
        Args:
            config: Application config.
            db_path: Path to the SQLite database.
            seeds_path: Path to seeds.yaml.
            failover_sources: Alternative routes to the *same* data, tried in
                order until one succeeds — Instagram via Apify, then Picuki,
                then Dumpor. Only the first success is used.
            independent_sources: Sources each covering different events, all of
                them fetched, one failing never stopping the others.
            logger: Structured logger.
            blocklist: Raw entries from data/blocklist.json.
        """
        self._config = config
        self._db_path = db_path
        self._seeds_path = seeds_path
        self._failover_sources = failover_sources
        self._independent_sources = independent_sources
        self._logger = logger
        self._blocklist = blocklist or []

    def run(
        self,
        get_now: Callable[[], datetime] = datetime.now,
        persist: bool = True,
        collect_raw: bool = False,
    ) -> IngestionResult:
        """Execute one ingestion pass.

        Args:
            get_now: Injectable time source.
            persist: Whether to write anything. `False` fetches and filters as
                normal but touches no table — a dry run must leave the database
                exactly as it found it, while still proving the providers work.
            collect_raw: Keep every candidate as fetched, with the reason any
                was discarded. Off by default because it holds the whole fetch
                in memory; on when a diagnostic run asks to dump it.

        Returns:
            IngestionResult with the accepted candidates and their counts.
        """
        conn = connect(self._db_path) if persist else None
        try:
            seeds = load_seeds(self._seeds_path)
            if conn is not None:
                self._sync_seeds(conn, seeds, get_now)
                conn.commit()

            seed_handles = {h for h in seeds.handles}
            pairs, failed_sources = self._collect_candidates()

            accepted: list[EventCandidate] = []
            pending_discovery: list[tuple[str, str]] = []
            discarded = 0
            handles_discovered = 0
            raw: list[RawCandidateRecord] = []
            fetched_by_source: dict[str, int] = {}
            accepted_by_source: dict[str, int] = {}

            for source_name, _ in pairs:
                fetched_by_source[source_name] = fetched_by_source.get(source_name, 0) + 1
            # A source that returned nothing has no pairs, and its silence is
            # the single most useful line in the report.
            for source in self._independent_sources:
                fetched_by_source.setdefault(source.source_name, 0)

            def _record(source_name: str, ec: EventCandidate, reason: str | None) -> None:
                if collect_raw:
                    raw.append(
                        RawCandidateRecord(
                            candidate=ec,
                            source=source_name,
                            verdict="accepted" if reason is None else "discarded",
                            reason=reason,
                        )
                    )

            extractor = HandleExtractor(
                db_path=self._db_path,
                max_depth=self._config.scraping.max_discovery_depth,
                blocklist=self._blocklist,
                logger=self._logger,
            )

            now = get_now()
            cutoff = now - timedelta(days=self._config.scraping.lookback_days)
            zone = _zone_of(self._config.location.timezone)
            floor = night_start(now, self._config.day_starts_at, zone)
            ceiling = floor + timedelta(days=self._config.scraping.horizon_days)

            for source_name, ec in pairs:
                if not self._within_event_window(ec, floor, ceiling, zone):
                    self._logger.info(
                        f"Discarding out-of-window event from {ec.source}: "
                        f"start_time={ec.start_time}",
                        component="ingestion",
                        duration_ms=0,
                    )
                    _record(source_name, ec, f"out of window: start_time={ec.start_time}")
                    discarded += 1
                    continue

                if not self._passes_lookback(ec, cutoff, now, zone):
                    self._logger.info(
                        f"Discarding old post from {ec.source}: raw_published_at={ec.raw_published_at}",
                        component="ingestion",
                        duration_ms=0,
                    )
                    _record(
                        source_name,
                        ec,
                        f"outside lookback: raw_published_at={ec.raw_published_at}",
                    )
                    discarded += 1
                    continue

                if self._is_malformed(ec):
                    self._logger.warning(
                        f"Discarding malformed candidate from {ec.source}: "
                        "title, description, and start_time are all absent",
                        component="ingestion",
                        duration_ms=0,
                    )
                    _record(
                        source_name,
                        ec,
                        "malformed: title, description and start_time are all absent",
                    )
                    discarded += 1
                    continue

                _record(source_name, ec, None)
                accepted_by_source[source_name] = accepted_by_source.get(source_name, 0) + 1
                accepted.append(ec)

                if conn is not None:
                    self._persist_candidate(conn, ec)
                    if ec.description:
                        pending_discovery.append((ec.description, ec.source))

            if conn is not None:
                # Commit before discovery, not after. HandleExtractor opens its
                # own connection, while the inserts above hold a RESERVED lock
                # until this commit — so running discovery inside the loop makes
                # it wait on a lock this same call stack is holding, and it dies
                # at SQLite's five-second timeout. It only bites when a
                # description actually mentions an @handle, since the extractor
                # returns before connecting when it finds none.
                conn.commit()

                for text, source_handle in pending_discovery:
                    extractor.process(
                        text=text, source_handle=source_handle, source_depth=0
                    )
                    handles_discovered += 1

                self._evaluate_promotion(conn, seed_handles, get_now)
                conn.commit()
        finally:
            if conn is not None:
                conn.close()

        return IngestionResult(
            accepted=len(accepted),
            discarded=discarded,
            handles_discovered=handles_discovered,
            candidates=accepted,
            per_source={
                name: SourceTally(fetched=count, accepted=accepted_by_source.get(name, 0))
                for name, count in fetched_by_source.items()
            },
            failed_sources=failed_sources,
            raw=raw,
        )

    # ------------------------------------------------------------------
    # Seed sync
    # ------------------------------------------------------------------

    def _sync_seeds(
        self,
        conn: sqlite3.Connection,
        seeds: Any,
        get_now: Callable[[], datetime],
    ) -> None:
        """Upsert seed handles into candidate_entities as active, depth=0."""
        now = get_now().isoformat()
        for handle in seeds.handles:
            existing = conn.execute(
                "SELECT id FROM candidate_entities WHERE handle = ?", (handle,)
            ).fetchone()
            if existing:
                conn.execute(
                    "UPDATE candidate_entities SET state = 'active', depth = 0, updated_at = ? WHERE handle = ?",
                    (now, handle),
                )
            else:
                conn.execute(
                    """INSERT INTO candidate_entities
                       (id, handle, state, depth, mention_count, mention_sources, created_at, updated_at)
                       VALUES (?, ?, 'active', 0, 0, '[]', ?, ?)""",
                    (str(uuid.uuid4()), handle, now, now),
                )

    # ------------------------------------------------------------------
    # Candidate collection
    # ------------------------------------------------------------------

    def _collect_candidates(self) -> tuple[list[tuple[str, EventCandidate]], list[str]]:
        """Run the failover chain plus every independent source, combined.

        The two lists differ by fetch policy, not by subject matter. Putting a
        source in the failover list means "only fetch this if the ones before
        it failed", which would silently starve any source that is not an
        alternative route to the same data.

        Returns:
            Each candidate paired with the name of the source that produced it,
            and the names of any sources that raised. The pairing is what lets a
            report say which of seventeen sources went quiet; `candidate.source`
            cannot, since several feeds can share one `source` value.
        """
        pairs: list[tuple[str, EventCandidate]] = []
        failed: list[str] = []

        # The chain reports only the source that succeeded, by design — the
        # others were never meant to run — so it is attributed as one unit.
        chain = FailoverChain(sources=self._failover_sources, logger=self._logger)
        if self._failover_sources:
            fetched = chain.fetch_all()
            pairs.extend(("failover_chain", candidate) for candidate in fetched)

        for source in self._independent_sources:
            name = source.source_name
            try:
                fetched = source.fetch()
            except Exception as exc:
                failed.append(name)
                self._logger.error(
                    f"Source {name} failed: {exc}",
                    component="ingestion",
                    duration_ms=0,
                )
                continue
            pairs.extend((name, candidate) for candidate in fetched)

        return pairs, failed

    # ------------------------------------------------------------------
    # Filtering
    # ------------------------------------------------------------------

    @staticmethod
    def _passes_lookback(
        ec: EventCandidate, cutoff: datetime, now: datetime, zone: tzinfo
    ) -> bool:
        """Decide whether a candidate is recent enough to ingest.

        The lookback exists to drop stale social posts, which is a judgement about
        an announcement's age. A candidate whose event has not happened yet is not
        stale under any reading, no matter how far back it was announced — forward
        looking sources such as public calendars routinely carry both.

        Every comparison localises first. Sources disagree about naivety —
        Do617 and the ICS feeds state an offset, HTML listings do not — and this
        filter and `_within_event_window` have to read a bare time the same way
        or one of them raises on input the other accepted.
        """
        if ec.start_time is not None and _localise(ec.start_time, zone) > _localise(now, zone):
            return True

        if ec.raw_published_at is None:
            return True
        return _localise(ec.raw_published_at, zone) >= _localise(cutoff, zone)

    @staticmethod
    def _within_event_window(
        ec: EventCandidate, floor: datetime, ceiling: datetime, zone: tzinfo
    ) -> bool:
        """Decide whether a candidate's own event time is worth ingesting.

        Two bounds, both on the event rather than the announcement. An event
        that is over cannot be attended, so keeping it is pure cost; an event
        past the horizon is beyond what this run is scoped to rank.

        A candidate with no start time is kept. A missing time is a gap in what
        we know rather than evidence about when, and the CLI renders those under
        their own heading — the window must not quietly take them.

        The lower bound uses `end_time` when there is one, so a run still under
        way is not discarded for having begun before tonight.
        """
        if ec.start_time is None:
            return True

        start = _localise(ec.start_time, zone)
        finish = _localise(ec.end_time, zone) if ec.end_time is not None else start

        return finish >= _localise(floor, zone) and start < _localise(ceiling, zone)

    @staticmethod
    def _is_malformed(ec: EventCandidate) -> bool:
        return ec.title is None and ec.description is None and ec.start_time is None

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _persist_candidate(self, conn: sqlite3.Connection, ec: EventCandidate) -> None:
        """Write one candidate through the shared row mapper.

        The column list lives with the reader (issue #22). Two hand-written
        lists in two modules drift silently — and did: `timing`, and later the
        authored summary and tags, reached the writer's table and never the
        reader's, so the batch reloaded them as defaults and preferred that copy.
        """
        write_candidates(conn, [ec])

    # ------------------------------------------------------------------
    # Promotion
    # ------------------------------------------------------------------

    def _evaluate_promotion(
        self,
        conn: sqlite3.Connection,
        seed_handles: set[str],
        get_now: Callable[[], datetime],
    ) -> None:
        """Promote probationary handles that meet the threshold from seed sources."""
        threshold = self._config.scraping.candidate_promotion_threshold
        now = get_now().isoformat()

        rows = conn.execute(
            """SELECT id, handle, mention_count, mention_sources
               FROM candidate_entities
               WHERE state = 'probationary' AND llm_classification = 'venue'""",
        ).fetchall()

        for entity_id, handle, mention_count, sources_json in rows:
            if mention_count < threshold:
                continue
            sources: list[str] = json.loads(sources_json) if sources_json else []
            has_seed_source = any(s in seed_handles for s in sources)
            if has_seed_source:
                conn.execute(
                    "UPDATE candidate_entities SET state = 'active', updated_at = ? WHERE id = ?",
                    (now, entity_id),
                )
                self._logger.info(
                    f"Promoted {handle} to active",
                    component="ingestion",
                    duration_ms=0,
                )


def _localise(value: datetime, zone: tzinfo) -> datetime:
    """Read a bare time as local, leaving one that states its own zone alone.

    Sources genuinely differ: Do617 and the ICS feeds carry an offset, HTML
    listings carry a wall clock with no zone at all. Both are legitimate, so
    every comparison has to normalise rather than assume.
    """
    return value if value.tzinfo is not None else value.replace(tzinfo=zone)


def _zone_of(name: str) -> ZoneInfo:
    """Resolve the configured zone, falling back to UTC rather than failing a run.

    Only used to place the night boundary and to localise a naive candidate, so
    a fallback shifts a boundary by hours rather than losing the batch.
    """
    try:
        return ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError):
        return ZoneInfo("UTC")
