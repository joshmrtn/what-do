"""IngestionService — orchestrates the full Phase 3 ingestion run."""

from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

from src.config import AppConfig
from src.ingestion.failover import FailoverChain
from src.ingestion.handle_extractor import HandleExtractor
from src.ingestion.seeds import load_seeds
from src.ingestion.source import IngestionSource
from src.models.event_candidate import EventCandidate


@dataclass
class IngestionResult:
    """Summary of a single ingestion run."""

    persisted: int
    discarded: int
    handles_discovered: int


class IngestionService:
    """Runs the full ingestion pipeline: seed load, scraping, filtering, persistence."""

    def __init__(
        self,
        config: AppConfig,
        db_path: Path,
        seeds_path: Path,
        social_sources: list[IngestionSource],
        movie_sources: list[IngestionSource],
        logger: Any,
        blocklist: list[str] | None = None,
    ) -> None:
        self._config = config
        self._db_path = db_path
        self._seeds_path = seeds_path
        self._social_sources = social_sources
        self._movie_sources = movie_sources
        self._logger = logger
        self._blocklist = blocklist or []

    def run(self, get_now: Callable[[], datetime] = datetime.now) -> IngestionResult:
        """Execute one ingestion pass.

        Args:
            get_now: Injectable time source.

        Returns:
            IngestionResult with counts of persisted and discarded candidates.
        """
        conn = sqlite3.connect(self._db_path)
        try:
            seeds = load_seeds(self._seeds_path)
            self._sync_seeds(conn, seeds, get_now)
            conn.commit()

            seed_handles = {h for h in seeds.handles}
            candidates = self._collect_candidates()

            persisted = 0
            discarded = 0
            handles_discovered = 0

            extractor = HandleExtractor(
                db_path=self._db_path,
                max_depth=self._config.scraping.max_discovery_depth,
                blocklist=self._blocklist,
                logger=self._logger,
            )

            now = get_now()
            cutoff = now - timedelta(days=self._config.scraping.lookback_days)

            for ec in candidates:
                if not self._passes_lookback(ec, cutoff):
                    self._logger.info(
                        f"Discarding old post from {ec.source}: raw_published_at={ec.raw_published_at}",
                        component="ingestion",
                        duration_ms=0,
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
                    discarded += 1
                    continue

                self._persist_candidate(conn, ec)
                persisted += 1

                if ec.description:
                    extractor.process(
                        text=ec.description,
                        source_handle=ec.source,
                        source_depth=0,
                    )
                    handles_discovered += 1

            conn.commit()
            self._evaluate_promotion(conn, seed_handles, get_now)
            conn.commit()
        finally:
            conn.close()

        return IngestionResult(
            persisted=persisted,
            discarded=discarded,
            handles_discovered=handles_discovered,
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

    def _collect_candidates(self) -> list[EventCandidate]:
        """Run social failover chain + all movie sources, combining results."""
        candidates: list[EventCandidate] = []

        social_chain = FailoverChain(sources=self._social_sources, logger=self._logger)
        candidates.extend(social_chain.fetch_all())

        for source in self._movie_sources:
            try:
                candidates.extend(source.fetch())
            except Exception as exc:
                self._logger.error(
                    f"Movie source {source.__class__.__name__} failed: {exc}",
                    component="ingestion",
                    duration_ms=0,
                )

        return candidates

    # ------------------------------------------------------------------
    # Filtering
    # ------------------------------------------------------------------

    @staticmethod
    def _passes_lookback(ec: EventCandidate, cutoff: datetime) -> bool:
        if ec.raw_published_at is None:
            return True
        return ec.raw_published_at >= cutoff

    @staticmethod
    def _is_malformed(ec: EventCandidate) -> bool:
        return ec.title is None and ec.description is None and ec.start_time is None

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _persist_candidate(self, conn: sqlite3.Connection, ec: EventCandidate) -> None:
        conn.execute(
            """INSERT OR REPLACE INTO event_candidates
               (id, source, source_type, url, image_url, raw_published_at,
                title, description, venue, location, start_time, end_time, discovered_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                ec.id,
                ec.source,
                ec.source_type,
                ec.url,
                ec.image_url,
                ec.raw_published_at.isoformat() if ec.raw_published_at else None,
                ec.title,
                ec.description,
                ec.venue,
                ec.location,
                ec.start_time.isoformat() if ec.start_time else None,
                ec.end_time.isoformat() if ec.end_time else None,
                ec.discovered_at.isoformat(),
            ),
        )

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
