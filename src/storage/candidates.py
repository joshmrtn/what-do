"""Row mapping for event candidates.

One column list, one pair of mapping functions, used by both the reader and the
writer. Issue #22: two hand-written column lists in two modules drift silently,
and the reader defaults the difference away — which is exactly how
`EventCandidate.timing` was dropped on every round trip for weeks.

Reading and writing themselves live in `storage/sqlite/candidates.py`.
"""

from __future__ import annotations

import hashlib
import json

from datetime import datetime
from typing import Any

from src.models.event_candidate import EventCandidate
from src.models.tag import Tag

#: Shared by the reader and the writer below, so a new field cannot reach one
#: without the other. Issue #22: two hand-written column lists in two modules
#: drift silently, and the reader defaults the difference away.
CANDIDATE_COLUMNS = (
    "id, source, source_type, url, image_url, raw_published_at, title, "
    "description, venue, location, start_time, end_time, discovered_at, "
    "timing, summary, tags, metadata, last_seen_at"
)


#: What a *source* published, as against what we recorded about seeing it.
#: `discovered_at` and `last_seen_at` are deliberately absent: they move on every
#: fetch, so including them would make each re-fetch look like an edit and the
#: version history would grow by the whole corpus nightly instead of by the
#: measured 2.26% that actually changes.
PUBLISHED_FIELDS = (
    "url",
    "image_url",
    "raw_published_at",
    "title",
    "description",
    "venue",
    "location",
    "start_time",
    "end_time",
    "timing",
    "summary",
    "tags",
)


def published_payload(candidate: EventCandidate) -> dict[str, Any]:
    """The candidate's published content, JSON-ready.

    One list feeds both this and `content_fingerprint`, so a field added to
    `EventCandidate` cannot end up hashed but unrecorded, or recorded but
    unhashed — either of which makes an edit invisible.
    """
    payload: dict[str, Any] = {}
    for name in PUBLISHED_FIELDS:
        value = getattr(candidate, name)
        if isinstance(value, datetime):
            payload[name] = value.isoformat()
        elif name == "tags":
            payload[name] = [{"tag": t.text, "weight": t.weight} for t in value]
        else:
            payload[name] = value
    return payload


def content_fingerprint(payload: dict[str, Any]) -> str:
    """Stable digest of a published payload.

    `sort_keys` so the digest does not depend on dict ordering, which would make
    an identical listing look edited on a Python upgrade.
    """
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


def row_to_candidate(row: tuple[Any, ...]) -> EventCandidate:
    """Rebuild an EventCandidate from a row of `CANDIDATE_COLUMNS`."""
    return EventCandidate(
        id=row[0],
        source=row[1],
        source_type=row[2],
        url=row[3],
        image_url=row[4],
        raw_published_at=_parse(row[5]),
        title=row[6],
        description=row[7],
        venue=row[8],
        location=row[9],
        start_time=_parse(row[10]),
        end_time=_parse(row[11]),
        discovered_at=datetime.fromisoformat(row[12]),
        timing=row[13],
        summary=row[14],
        tags=[Tag(text=t["tag"], weight=t["weight"]) for t in json.loads(row[15] or "[]")],
        metadata=json.loads(row[16] or "{}"),
        # Not defaulted to `discovered_at` when absent: a stored row that lost
        # its last sighting would then read as a first sighting, which is the
        # conflation the column exists to end. Unreachable after the migration
        # backfills, and loud rather than quiet if it ever is.
        last_seen_at=datetime.fromisoformat(row[17]),
    )


def candidate_to_row(candidate: EventCandidate) -> tuple[Any, ...]:
    """Flatten a candidate into a row of `CANDIDATE_COLUMNS`."""
    return (
        candidate.id,
        candidate.source,
        candidate.source_type,
        candidate.url,
        candidate.image_url,
        candidate.raw_published_at.isoformat() if candidate.raw_published_at else None,
        candidate.title,
        candidate.description,
        candidate.venue,
        candidate.location,
        candidate.start_time.isoformat() if candidate.start_time else None,
        candidate.end_time.isoformat() if candidate.end_time else None,
        candidate.discovered_at.isoformat(),
        candidate.timing,
        candidate.summary,
        json.dumps([{"tag": t.text, "weight": t.weight} for t in candidate.tags]),
        json.dumps(candidate.metadata),
        (candidate.last_seen_at or candidate.discovered_at).isoformat(),
    )


def _parse(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value) if value else None
