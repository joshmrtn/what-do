"""User preference loading, domain scoping, and embedding cache.

Preference files change rarely, so their embeddings are cached per line in
`preference_embeddings_cache` and only regenerated for lines that actually
changed. Lines are keyed by a hash of their text, which is what the embedding
depends on — moving a line between domain sections updates the stored domain
without paying for a new embedding.
"""

from __future__ import annotations

import hashlib
import sqlite3

from src.storage.db import connect
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from src.scoring.embeddings import EmbeddingError, EmbeddingProvider
from src.utils.logging import StructuredLogger
from src.utils.text import normalize_embedding_text
from src.utils.vectors import decode_vector, encode_vector

GENERAL_DOMAIN = "general"
_COMMENT_PREFIX = "#"

#: Longest accepted preference line. A preference is a short phrase; anything
#: near this is a paste accident. Measured: nomic-embed-text swallows emoji,
#: control characters, and 10k-word inputs without complaint, but a 100k-char
#: line times out — which, with fatal embedding failures, would wedge every
#: subsequent batch run until the file was fixed.
MAX_PREFERENCE_LENGTH = 500


class PreferenceError(ValueError):
    """Raised when a preference file cannot be read or contains an unusable line."""


@dataclass
class UserPreference:
    """A single preference line with its domain and vector.

    Args:
        preference_type: "like" or "dislike".
        domain: Section it appeared under; "general" applies to every event.
        text: The line as written in the file.
        embedding: Vector for text, empty until the cache populates it.
    """

    preference_type: str
    domain: str
    text: str
    embedding: list[float] = field(default_factory=list)


@dataclass
class PreferenceSet:
    """Loaded likes and dislikes, ready for scoring."""

    likes: list[UserPreference] = field(default_factory=list)
    dislikes: list[UserPreference] = field(default_factory=list)

    def likes_for(self, domain: str) -> list[UserPreference]:
        """Likes applicable to an event in the given domain."""
        return _scoped(self.likes, domain)

    def dislikes_for(self, domain: str) -> list[UserPreference]:
        """Dislikes applicable to an event in the given domain."""
        return _scoped(self.dislikes, domain)


def _scoped(prefs: list[UserPreference], domain: str) -> list[UserPreference]:
    """General preferences always apply; domain ones only to their own domain."""
    return [p for p in prefs if p.domain in (GENERAL_DOMAIN, domain)]


def parse_preferences(content: str, preference_type: str) -> list[UserPreference]:
    """Parse a preference file into domain-scoped preferences.

    Lines before the first `[section]` header belong to the general domain.
    Blank lines and `#` comments are skipped, as are lines left empty once
    zero-signal characters (emoji, zero-width marks) are stripped.

    Args:
        content: Raw file contents.
        preference_type: "like" or "dislike".

    Returns:
        Preferences in file order, without embeddings.

    Raises:
        PreferenceError: If a line exceeds MAX_PREFERENCE_LENGTH.
    """
    prefs: list[UserPreference] = []
    domain = GENERAL_DOMAIN

    for raw_line in content.splitlines():
        line = normalize_embedding_text(raw_line)
        if not line or line.startswith(_COMMENT_PREFIX):
            continue
        if line.startswith("[") and line.endswith("]"):
            domain = line[1:-1].strip().lower()
            continue
        if len(line) > MAX_PREFERENCE_LENGTH:
            raise PreferenceError(
                f"Preference line is too long "
                f"({len(line)} chars, limit {MAX_PREFERENCE_LENGTH}): {line[:60]!r}..."
            )
        prefs.append(
            UserPreference(preference_type=preference_type, domain=domain, text=line)
        )

    return prefs


def _line_hash(text: str) -> str:
    """Stable key for a preference line — the embedding depends on text alone."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class PreferenceRepository:
    """Loads preference files, embedding only what changed since the last run.

    Args:
        provider: Embedding provider.
        db_path: Path to the SQLite database holding the cache.
        logger: Structured logger for missing files and embedding failures.
    """

    def __init__(
        self,
        provider: EmbeddingProvider,
        db_path: Path | str,
        logger: StructuredLogger,
    ) -> None:
        self._provider = provider
        self._db_path = Path(db_path)
        self._logger = logger

    def load(self, likes_path: Path | str, dislikes_path: Path | str) -> PreferenceSet:
        """Load both preference files, using cached embeddings where valid.

        Args:
            likes_path: Path to likes.txt.
            dislikes_path: Path to dislikes.txt.

        Returns:
            PreferenceSet with embeddings populated. Lines whose embedding
            could not be generated are omitted and logged.
        """
        conn = connect(self._db_path)
        try:
            try:
                likes = self._load_file(conn, Path(likes_path), "like")
                dislikes = self._load_file(conn, Path(dislikes_path), "dislike")
            except EmbeddingError:
                # Keep whatever embedded successfully before the failure. Cache
                # rows are content-addressed, so a partial cache is never wrong,
                # only incomplete — and it makes the retry cheap.
                conn.commit()
                raise
            conn.commit()
        finally:
            conn.close()

        return PreferenceSet(likes=likes, dislikes=dislikes)

    def _load_file(
        self, conn: sqlite3.Connection, path: Path, preference_type: str
    ) -> list[UserPreference]:
        """Parse one preference file and attach embeddings from cache or provider."""
        if not path.exists():
            # Absent means "none configured" — a legitimate first-run state.
            self._logger.warning(
                f"Preference file not found, treating as empty: {path}",
                component="preferences",
                duration_ms=0,
            )
            return []

        try:
            content = path.read_text()
        except (OSError, UnicodeDecodeError) as exc:
            # Present but unreadable means real preferences are being ignored,
            # which silently changes every score. Refuse rather than degrade.
            raise PreferenceError(
                f"Preference file {path} exists but could not be read: {exc}"
            ) from exc

        try:
            prefs = _deduplicate(parse_preferences(content, preference_type))
        except PreferenceError as exc:
            raise PreferenceError(f"In {path.name}: {exc}") from exc
        file_name = path.name
        cached = self._read_cache(conn, file_name)

        resolved: list[UserPreference] = []
        for pref in prefs:
            key = _line_hash(pref.text)
            row = cached.get(key)

            if row is not None:
                pref.embedding = decode_vector(row[0])
                if row[1] != pref.domain:
                    self._update_domain(conn, file_name, key, pref.domain)
            else:
                try:
                    # Quantise through float32 storage immediately so a cold
                    # run and a cached run yield bit-identical vectors, and
                    # therefore bit-identical scores.
                    pref.embedding = decode_vector(
                        encode_vector(self._provider.embed(pref.text))
                    )
                except EmbeddingError as exc:
                    # Fatal by design. Skipping the line would score the whole
                    # batch against a partial preference set — losing a dislike
                    # silently lifts every event's score, with nothing in the
                    # output to show it happened. A loud failure is recoverable;
                    # a plausible-looking wrong ranking is not.
                    self._logger.error(
                        f"Embedding failed for preference {pref.text!r} "
                        f"in {file_name}: {exc}",
                        component="preferences",
                        duration_ms=0,
                    )
                    raise EmbeddingError(
                        f"Cannot score against a partial preference set: "
                        f"embedding failed for {pref.text!r} in {file_name} ({exc})"
                    ) from exc
                self._insert(conn, file_name, key, pref)

            resolved.append(pref)

        self._prune(conn, file_name, {_line_hash(p.text) for p in resolved})
        return resolved

    @staticmethod
    def _read_cache(
        conn: sqlite3.Connection, file_name: str
    ) -> dict[str, tuple[bytes, str]]:
        """Return {line_hash: (embedding_blob, domain)} for one file."""
        rows = conn.execute(
            "SELECT line_hash, embedding, domain FROM preference_embeddings_cache "
            "WHERE file_name = ?",
            (file_name,),
        ).fetchall()
        return {r[0]: (r[1], r[2]) for r in rows}

    @staticmethod
    def _insert(
        conn: sqlite3.Connection, file_name: str, key: str, pref: UserPreference
    ) -> None:
        conn.execute(
            """
            INSERT OR REPLACE INTO preference_embeddings_cache
                (id, file_name, line_hash, line_text, domain,
                 preference_type, embedding, generated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(uuid.uuid4()),
                file_name,
                key,
                pref.text,
                pref.domain,
                pref.preference_type,
                encode_vector(pref.embedding),
                datetime.now(timezone.utc).isoformat(),
            ),
        )

    @staticmethod
    def _update_domain(
        conn: sqlite3.Connection, file_name: str, key: str, domain: str
    ) -> None:
        conn.execute(
            "UPDATE preference_embeddings_cache SET domain = ? "
            "WHERE file_name = ? AND line_hash = ?",
            (domain, file_name, key),
        )

    @staticmethod
    def _prune(conn: sqlite3.Connection, file_name: str, keep: set[str]) -> None:
        """Drop cache rows for lines no longer present in the file."""
        rows = conn.execute(
            "SELECT line_hash FROM preference_embeddings_cache WHERE file_name = ?",
            (file_name,),
        ).fetchall()
        stale = [(file_name, r[0]) for r in rows if r[0] not in keep]
        if stale:
            conn.executemany(
                "DELETE FROM preference_embeddings_cache "
                "WHERE file_name = ? AND line_hash = ?",
                stale,
            )


def _deduplicate(prefs: list[UserPreference]) -> list[UserPreference]:
    """Collapse repeated lines within a file, keeping the first occurrence."""
    seen: set[str] = set()
    unique: list[UserPreference] = []
    for pref in prefs:
        if pref.text in seen:
            continue
        seen.add(pref.text)
        unique.append(pref)
    return unique
