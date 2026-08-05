"""Shared blocklist matching.

`data/blocklist.json` mixes two kinds of entry: `@handles`, which identify a
source exactly, and venue names, which are typed by hand and so are matched
fuzzily. Both discovery and ranking consult it, and they must agree — a venue
blocked at discovery that still scored at ranking would be a blocklist that
only works on things you have not seen yet.
"""

from __future__ import annotations

from collections.abc import Iterable

from rapidfuzz import fuzz

#: Blocklist entries starting with this are matched against handles, not names.
HANDLE_PREFIX = "@"


def name_similarity(a: str, b: str) -> float:
    """Return the rapidfuzz ratio (0-100) between two names, case-insensitively."""
    return float(fuzz.ratio(a.lower(), b.lower()))


def is_blocked(
    name: str | None,
    handles: Iterable[str],
    blocklist: Iterable[str],
    threshold: float,
) -> bool:
    """Whether a venue is on the blocklist.

    Args:
        name: Venue name, if known. A missing or blank name can still be
            blocked by handle, but never matches a name entry — an empty string
            scores against every entry and would block indiscriminately.
        handles: Social handles associated with the venue.
        blocklist: Raw entries from `data/blocklist.json`.
        threshold: Name match threshold in 0.0-1.0.

    Returns:
        True if any entry matches.
    """
    handle_set = set(handles)
    clean_name = name.strip() if name else ""

    for entry in blocklist:
        if entry.startswith(HANDLE_PREFIX):
            if entry in handle_set:
                return True
        elif clean_name and name_similarity(clean_name, entry) >= threshold * 100:
            return True

    return False
