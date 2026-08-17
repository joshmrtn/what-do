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
from pathlib import Path

from src.models.preference_revision import PreferenceLine, PreferenceRevision
from src.scoring.preferences import (
    PreferenceError,
    PreferenceSet,
    UserPreference,
    parse_preferences,
)

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


def hash_preference_files(likes_path: Path, dislikes_path: Path) -> str:
    """Hash the preference files as they stand, embedding nothing.

    The read path's half of the comparison. It parses but never embeds, so
    asking whether preferences have moved costs a file read rather than a model
    call — the question must be cheap enough to ask on every invocation.

    An unreadable or absent file is treated as empty rather than raised on: a
    first run has neither, and a listing must never be lost to a question it
    only asked out of helpfulness.
    """

    def parsed(path: Path, preference_type: str) -> list[UserPreference]:
        try:
            return parse_preferences(path.read_text(), preference_type)
        except (OSError, UnicodeDecodeError, PreferenceError):
            return []

    return content_hash_of(
        PreferenceSet(
            likes=parsed(likes_path, "like"),
            dislikes=parsed(dislikes_path, "dislike"),
        ),
        likes_name=likes_path.name,
        dislikes_name=dislikes_path.name,
    )


def content_hash_of(
    preferences: PreferenceSet, *, likes_name: str, dislikes_name: str
) -> str:
    """The hash alone, for a caller that only wants to compare.

    The read path asks "have the files moved since the batch scored them?" and
    has no business inventing a capture time to find out — a revision it never
    intends to record would be a lie about when this content was first seen.
    """
    return content_hash(
        _lines_of(preferences.likes, likes_name)
        + _lines_of(preferences.dislikes, dislikes_name)
    )


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
