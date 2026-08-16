"""The short, copyable name an event is known by on screen.

A rank cannot be that name. It is re-derived every night, and once displayed
numbering is view-local it names a different event in every view — so a reader
who types the number they counted gets a valid explanation of the wrong event.
The handle is derived from `event_id`, which is stable across runs (measured
2026-08-16: 530 events created on 08-09 were still ranked under the same id a
week later), so it means the same thing in `--raw`, in a filtered listing, and
in yesterday's scrollback.

**Derived, not sliced.** Ids are not uniformly UUIDs — a synthetic event is
keyed on its rule and date, so `event_id[:7]` reads `synthet` for every one of
them. Hashing gives a uniform width whatever the id scheme.
"""

from __future__ import annotations

import hashlib

#: Seven hex characters. At 1935 events the birthday probability of any
#: collision is about 0.7%, and a collision degrades to a question rather than a
#: wrong answer: `matching` returns every hit and makes the caller disambiguate.
HANDLE_LENGTH = 7

#: Marks the handle as a handle. Without it a bare hex string that happens to be
#: all digits is indistinguishable from a position in the listing, which is the
#: ambiguity the handle exists to remove.
HANDLE_SIGIL = "#"


def short_handle(event_id: str) -> str:
    """The handle for one event, without its sigil.

    Args:
        event_id: The event's stable id, in any of the schemes in use.

    Returns:
        Seven lowercase hex characters.
    """
    return hashlib.sha1(event_id.encode()).hexdigest()[:HANDLE_LENGTH]
