"""A snapshot of the preference files that produced a ranking.

Scores are only meaningful against the preferences they were computed from, and
`likes.txt` is gitignored and edited freely. Without this, a score whose
preferences have since changed is unexplainable — the number survives and what
it was measured against does not.

The revision carries no id. Identity belongs to storage, which reuses the row
for content it has already seen, so an unedited file does not mint a revision
every night.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass(frozen=True)
class PreferenceLine:
    """One line of one preference file, as it stood at capture.

    Attributes:
        file_name: Which file it came from, by name rather than by path — the
            path is machine-local and the name is what identifies the list.
        position: Where it sat in the file, counted per file from zero.
        domain: The `[section]` it appeared under; "general" applies to every
            event.
        preference_type: "like" or "dislike".
        line_text: The line as parsed, after zero-signal characters are stripped.
        line_hash: Digest of `line_text`, the same key the embedding cache uses,
            so a stored line can be joined to the vector it was scored with.
    """

    file_name: str
    position: int
    domain: str
    preference_type: str
    line_text: str
    line_hash: str


@dataclass(frozen=True)
class PreferenceRevision:
    """What both preference files said at one moment.

    Attributes:
        captured_at: When this content was first seen. Deliberately not "last
            seen": the revision records when the preferences *became* this, and
            re-recording an unchanged file keeps the original stamp.
        content_hash: Digest over every line, its order, its domain and its
            list. Two runs agree on this exactly when nothing that scores has
            changed.
        lines: Every line from both files, in file order.
    """

    captured_at: datetime
    content_hash: str
    lines: list[PreferenceLine] = field(default_factory=list)
