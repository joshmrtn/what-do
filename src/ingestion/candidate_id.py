"""Stable candidate id derivation, shared by the ingestion adapters."""

from __future__ import annotations

import hashlib
from datetime import datetime

from src.normalization.deduplicator import canonical_title, canonical_venue

_DIGEST_LENGTH = 16

#: A listing's identity independent of whatever id the publisher attached to it.
ContentKey = tuple[str, str, str, str]


def content_identity(
    *,
    source: str,
    title: str | None,
    venue: str | None,
    start: datetime | None,
) -> ContentKey:
    """What identifies a listing, rather than the publisher's handle on it.

    **One rule, two callers.** `id_churn` uses this to decide whether a listing
    has been seen before; a source latched to content ids derives the id from
    it. A second copy of the rule would diverge silently and perfectly — the
    detector reporting 0% churn while the ids churned, because each answers with
    its own key.

    Scoped to the **feed**, not the category: a calendar feed and a listing page
    legitimately cover the same event, so a key spanning `source_type` would let
    one feed's stored row make the other's first publication look already known.

    Canonical on both text fields, because a re-cased republish would otherwise
    read as a brand new listing — the same reason dedup compares on a canonical
    key rather than raw strings.

    Carries the **start**: every occurrence of a recurring programme shares a
    title and venue, so a key without it collapses a whole season into one
    listing.
    """
    return (
        source,
        canonical_title(title or ""),
        canonical_venue(venue or ""),
        start.isoformat() if start else "",
    )


def identifies_a_listing(key: ContentKey) -> bool:
    """Whether this key says enough to tell one listing from another.

    A venue alone does not. A social post carries neither a title nor a start —
    extraction derives those later — so every post from one account collapses
    onto `(handle, "", location, "")`. Treated as a listing key that would read
    as total churn on a healthy source, and would key an entire account's feed
    onto one id if anything acted on it.

    A title *or* a start is enough: plenty of feeds publish a date they cannot
    pin to an hour, and plenty publish a title with no time at all.
    """
    _, title, _, start = key
    return bool(title or start)


def derive_content_id(
    *,
    source: str,
    title: str | None,
    venue: str | None,
    start: datetime | None,
) -> str:
    """A candidate id derived from the listing itself, not from its publisher.

    For a source whose identifiers have been shown not to identify anything.
    Keyed on `content_identity`, so it collapses exactly the listings the churn
    detector counts as one.
    """
    return derive_candidate_id(source, *content_identity(
        source=source, title=title, venue=venue, start=start
    ))


def derive_candidate_id(source_type: str, *parts: object) -> str:
    """Derive a candidate id that is identical on every fetch of the same item.

    Prefer the source's own identifier as the material. Content is a fallback for
    sources that publish none: it is stable across fetches, but an edit upstream
    mints a new id and re-extracts, which an identifier would not.

    Args:
        source_type: The adapter's source_type, used as a readable prefix.
        *parts: Material identifying the upstream item. `None` holds its position
            so an absent field cannot shift the remaining values into it.

    Returns:
        An id of the form ``"<source_type>:<16 hex chars>"``.

    Raises:
        ValueError: If no part carries content, which would otherwise collapse
            every such item onto one id.
    """
    rendered = [_render(part) for part in parts]
    if not any(rendered):
        raise ValueError(f"no identifying material for a {source_type} candidate")
    digest = hashlib.sha256("|".join(rendered).encode("utf-8")).hexdigest()
    return f"{source_type}:{digest[:_DIGEST_LENGTH]}"


def _render(part: object) -> str:
    if part is None:
        return ""
    if isinstance(part, datetime):
        return part.isoformat()
    return str(part)
