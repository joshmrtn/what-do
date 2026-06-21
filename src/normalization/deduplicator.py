"""Deduplication engine — Pass 1 (fuzzy, pre-embedding)."""

from __future__ import annotations

from dataclasses import replace
from datetime import timedelta

from rapidfuzz import fuzz

from src.config import DeduplicationConfig
from src.models.event import Event


def _title_match(a: str | None, b: str | None, threshold: float) -> bool:
    """True when titles are considered the same under the given threshold."""
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    return fuzz.token_sort_ratio(a, b) / 100.0 >= threshold


def _venue_match(a: str | None, b: str | None) -> bool:
    """True when venues are considered the same (exact canonical match)."""
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    return a == b


def _time_match(a_time, b_time, window_hours: float) -> bool:
    """True when start times are within the configured window."""
    if a_time is None and b_time is None:
        return True
    if a_time is None or b_time is None:
        return False
    return abs((a_time - b_time).total_seconds()) <= window_hours * 3600


def _are_duplicates(a: Event, b: Event, cfg: DeduplicationConfig) -> bool:
    return (
        _title_match(a.title, b.title, cfg.fuzzy_title_threshold)
        and _venue_match(a.venue, b.venue)
        and _time_match(a.start_time, b.start_time, cfg.time_window_hours)
    )


def _null_count(event: Event) -> int:
    """Count None-valued optional fields — lower means more complete."""
    fields = [
        event.url, event.image_url, event.title, event.venue,
        event.description, event.location, event.start_time, event.end_time,
        event.summary, event.summary_embedding, event.weather, event.astronomical_data,
    ]
    return sum(1 for f in fields if f is None)


def _merge(events: list[Event]) -> Event:
    """Merge a cluster of duplicate events into one canonical record.

    Most-complete record (fewest None fields) is the base. Tiebreak: earliest
    created_at. Non-None fields from secondary records fill gaps in the base.
    Source candidate IDs are unioned across all contributors.
    """
    ranked = sorted(events, key=lambda e: (_null_count(e), e.created_at))
    base = ranked[0]

    merged_candidates: list[str] = []
    for e in events:
        for cid in e.source_event_candidates:
            if cid not in merged_candidates:
                merged_candidates.append(cid)

    url = base.url
    image_url = base.image_url
    title = base.title
    venue = base.venue
    description = base.description
    location = base.location
    start_time = base.start_time
    end_time = base.end_time
    summary = base.summary
    summary_embedding = base.summary_embedding
    weather = base.weather
    astronomical_data = base.astronomical_data

    for secondary in ranked[1:]:
        if url is None:
            url = secondary.url
        if image_url is None:
            image_url = secondary.image_url
        if title is None:
            title = secondary.title
        if venue is None:
            venue = secondary.venue
        if description is None:
            description = secondary.description
        if location is None:
            location = secondary.location
        if start_time is None:
            start_time = secondary.start_time
        if end_time is None:
            end_time = secondary.end_time
        if summary is None:
            summary = secondary.summary
        if summary_embedding is None:
            summary_embedding = secondary.summary_embedding
        if weather is None:
            weather = secondary.weather
        if astronomical_data is None:
            astronomical_data = secondary.astronomical_data

    return replace(
        base,
        source_event_candidates=merged_candidates,
        url=url,
        image_url=image_url,
        title=title,
        venue=venue,
        description=description,
        location=location,
        start_time=start_time,
        end_time=end_time,
        summary=summary,
        summary_embedding=summary_embedding,
        weather=weather,
        astronomical_data=astronomical_data,
    )


class DeduplicationEngine:
    """Deduplicate a list of Events using fuzzy matching (Pass 1).

    Pure — no I/O, no DB access. Uses a greedy union-find approach:
    events are assigned to clusters; each cluster merges into one canonical event.
    """

    def deduplicate(self, events: list[Event], config: DeduplicationConfig) -> list[Event]:
        """Merge duplicate events within the given list.

        Args:
            events: Normalized events from NormalizationEngine.
            config: Deduplication thresholds and windows.

        Returns:
            Deduplicated list of Events (order not guaranteed).
        """
        if not events:
            return []

        n = len(events)
        parent = list(range(n))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x: int, y: int) -> None:
            px, py = find(x), find(y)
            if px != py:
                parent[py] = px

        for i in range(n):
            for j in range(i + 1, n):
                if _are_duplicates(events[i], events[j], config):
                    union(i, j)

        clusters: dict[int, list[Event]] = {}
        for i, event in enumerate(events):
            root = find(i)
            clusters.setdefault(root, []).append(event)

        return [_merge(cluster) for cluster in clusters.values()]
