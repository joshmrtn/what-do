"""SQLite-backed `PreferenceRevisionRepository`.

Writes the two tables the additive DDL pass created and nothing has used since:
`preference_revisions` and `preference_lines`.
"""

from __future__ import annotations

import sqlite3
import uuid
from datetime import datetime
from pathlib import Path

from src.models.preference_revision import PreferenceLine, PreferenceRevision
from src.storage.sqlite.connection import connect, transaction

_LINE_COLUMNS = (
    "revision_id, file_name, position, domain, preference_type, line_text, line_hash"
)


class SqlitePreferenceRevisionRepository:
    """Reads and writes preference revisions against a SQLite database."""

    def __init__(self, db_path: Path | str) -> None:
        self._db_path = db_path

    def record(self, revision: PreferenceRevision) -> str:
        """Store a revision, or recognise one already stored, returning its id.

        `content_hash` is UNIQUE, so an unedited preference file resolves to the
        row it already has rather than writing a new one every night. The lookup
        comes first because the reused row must keep its **original**
        `captured_at`: the revision records when the preferences became this,
        not when they were last seen.
        """
        conn = connect(self._db_path)
        try:
            existing = conn.execute(
                "SELECT id FROM preference_revisions WHERE content_hash = ?",
                (revision.content_hash,),
            ).fetchone()
            if existing is not None:
                return str(existing[0])

            revision_id = str(uuid.uuid4())
            # One transaction: a revision whose lines failed to write would be
            # indistinguishable from a genuinely empty preference file.
            with transaction(conn):
                conn.execute(
                    "INSERT INTO preference_revisions (id, captured_at, content_hash) "
                    "VALUES (?, ?, ?)",
                    (
                        revision_id,
                        revision.captured_at.isoformat(),
                        revision.content_hash,
                    ),
                )
                conn.executemany(
                    f"INSERT INTO preference_lines ({_LINE_COLUMNS}) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    [
                        (
                            revision_id,
                            line.file_name,
                            line.position,
                            line.domain,
                            line.preference_type,
                            line.line_text,
                            line.line_hash,
                        )
                        for line in revision.lines
                    ],
                )
            return revision_id
        finally:
            conn.close()

    def get(self, revision_id: str) -> PreferenceRevision | None:
        """One revision by id, with its lines in file order, or None."""
        conn = connect(self._db_path)
        try:
            row = conn.execute(
                "SELECT captured_at, content_hash FROM preference_revisions WHERE id = ?",
                (revision_id,),
            ).fetchone()
            if row is None:
                return None
            return self._with_lines(conn, revision_id, row)
        finally:
            conn.close()

    def latest(self) -> PreferenceRevision | None:
        """The most recently captured revision, or None if there are none.

        Ordered by `captured_at` rather than by insertion, because re-recording
        an older revision reuses its row and must not promote it to newest.
        """
        conn = connect(self._db_path)
        try:
            row = conn.execute(
                "SELECT id, captured_at, content_hash FROM preference_revisions "
                "ORDER BY captured_at DESC, id DESC LIMIT 1"
            ).fetchone()
            if row is None:
                return None
            return self._with_lines(conn, str(row[0]), (row[1], row[2]))
        finally:
            conn.close()

    @staticmethod
    def _with_lines(
        conn: sqlite3.Connection, revision_id: str, row: tuple[str, str]
    ) -> PreferenceRevision:
        """Attach a revision's lines, ordered as they sat in their files."""
        lines = conn.execute(
            f"SELECT {_LINE_COLUMNS} FROM preference_lines WHERE revision_id = ? "
            "ORDER BY file_name, position",
            (revision_id,),
        ).fetchall()
        return PreferenceRevision(
            captured_at=datetime.fromisoformat(row[0]),
            content_hash=row[1],
            lines=[
                PreferenceLine(
                    file_name=line[1],
                    position=line[2],
                    domain=line[3],
                    preference_type=line[4],
                    line_text=line[5],
                    line_hash=line[6],
                )
                for line in lines
            ],
        )
