"""Turn a loaded preference set into a revision that can be recorded.

Kept apart from `preferences.py`, which owns loading and the embedding cache.
This module answers a different question — *what did the files say?* — and it
must stay free of anything to do with vectors, because a revision that moved
when the embedding model changed would report a preference edit that never
happened.
"""

from __future__ import annotations

import hashlib
from datetime import datetime

from src.models.preference_revision import PreferenceLine, PreferenceRevision
from src.scoring.preferences import PreferenceSet, UserPreference

#: Separates the fields of one line, and one line from the next. Control
#: characters rather than punctuation, because a preference line may contain any
#: printable character and a separator it could contain is a collision waiting
#: to be typed.
_FIELD = "\x1f"
_RECORD = "\x1e"


def _line_hash(text: str) -> str:
    """Digest of a line's text — the same key the embedding cache stores under."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _lines_of(
    preferences: list[UserPreference], file_name: str
) -> list[PreferenceLine]:
    """One file's preferences as recordable lines, numbered from zero."""
    return [
        PreferenceLine(
            file_name=file_name,
            position=position,
            domain=preference.domain,
            preference_type=preference.preference_type,
            line_text=preference.text,
            line_hash=_line_hash(preference.text),
        )
        for position, preference in enumerate(preferences)
    ]


def content_hash(lines: list[PreferenceLine]) -> str:
    """Digest over everything about these lines that can change a score.

    Covers the text, the list it is on, the domain that scopes it, and the
    position — the last because the revision claims to describe the file, and
    calling a reordered file the same one would make the stored positions lie.

    Embeddings are deliberately absent. A vector is a pure function of text and
    model, so folding one in would mint a revision every time the model changed
    while the user's preferences stood still.
    """
    joined = _RECORD.join(
        _FIELD.join(
            (
                line.file_name,
                str(line.position),
                line.preference_type,
                line.domain,
                line.line_text,
            )
        )
        for line in lines
    )
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def build_revision(
    preferences: PreferenceSet,
    *,
    likes_name: str,
    dislikes_name: str,
    captured_at: datetime,
) -> PreferenceRevision:
    """Snapshot a loaded preference set.

    Args:
        preferences: The set as loaded, with or without embeddings attached.
        likes_name: Name of the likes file, for the stored lines.
        dislikes_name: Name of the dislikes file.
        captured_at: When this was read.

    Returns:
        A revision ready to record. Storage decides whether it is new.
    """
    lines = _lines_of(preferences.likes, likes_name) + _lines_of(
        preferences.dislikes, dislikes_name
    )
    return PreferenceRevision(
        captured_at=captured_at,
        content_hash=content_hash(lines),
        lines=lines,
    )
