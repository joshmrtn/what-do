"""HandleExtractor — extracts @handles from post text and records mentions.

The accumulation rules (a source counts once, the first context wins) live with
the repository, so this module is only about finding handles in text.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any, Callable

from src.storage.protocols import EntityRepository


def _default_now() -> datetime:
    return datetime.now(timezone.utc)


_HANDLE_RE = re.compile(r"@[\w.]+")


class HandleExtractor:
    """Parses @handles from post captions and upserts them into candidate_entities."""

    def __init__(
        self,
        entities: EntityRepository,
        max_depth: int,
        blocklist: list[str],
        logger: Any,
        get_now: Callable[[], datetime] = _default_now,
    ) -> None:
        self._entities = entities
        self._get_now = get_now
        self._max_depth = max_depth
        self._blocklist = {h.lower() for h in blocklist if h.startswith("@")}
        self._logger = logger

    def process(self, text: str, source_handle: str, source_depth: int) -> None:
        """Extract handles from text and persist new discoveries.

        Args:
            text: Post caption or description to scan.
            source_handle: The handle that published this text (for mention_sources tracking).
            source_depth: Discovery depth of the source handle.
        """
        candidate_depth = source_depth + 1
        if candidate_depth > self._max_depth:
            return

        handles = _HANDLE_RE.findall(text)
        if not handles:
            return

        context = text[:300]
        now = self._get_now()
        for handle in handles:
            if handle.lower() in self._blocklist:
                self._logger.info(
                    f"Skipping blocklisted handle: {handle}",
                    component="handle_extractor",
                    duration_ms=0,
                )
                continue
            self._entities.record_mention(
                handle=handle,
                source_handle=source_handle,
                depth=candidate_depth,
                context=context,
                now=now,
            )
