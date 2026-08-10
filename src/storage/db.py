"""SQLite database initialisation and schema management."""

from __future__ import annotations

import itertools
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

DEFAULT_DB_PATH = Path("database/event_hub.db")

#: How long a statement waits for a lock before giving up. The batch writes for
#: hours while the CLI reads, and without this a read landing on a checkpoint
#: fails instantly with "database is locked" rather than waiting a moment.
BUSY_TIMEOUT_MS = 5000


def connect(db_path: Path | str) -> sqlite3.Connection:
    """Open a connection carrying the settings this database depends on.

    Every connection must apply these, so nothing opens `sqlite3.connect`
    directly: `foreign_keys` and `busy_timeout` are per-connection and silently
    default to off and zero. `journal_mode` is a property of the file and is set
    once by `init_db`.

    Args:
        db_path: Path to the SQLite file.

    Returns:
        An open connection with foreign keys enforced and a busy timeout set.
    """
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute(f"PRAGMA busy_timeout = {BUSY_TIMEOUT_MS}")
    return conn


#: Savepoint names must be unique while nested, so each block takes the next one
#: rather than a fixed name a nested block would shadow.
_savepoint_names = itertools.count()


@contextmanager
def transaction(conn: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    """Group writes so they commit together or not at all.

    Built on `SAVEPOINT` rather than `BEGIN` for two reasons. Nesting works
    without bookkeeping — SQLite commits only when the outermost block releases,
    so a caller need not know whether it is already inside a transaction. And a
    savepoint opens a transaction implicitly, which keeps the driver from
    emitting its own `BEGIN` and failing with "cannot start a transaction within
    a transaction".

    Args:
        conn: An open connection, from `connect`.

    Yields:
        The same connection, for writes that should share the block's fate.

    Raises:
        Exception: Whatever the block raised, after rolling its writes back.
    """
    name = f"what_do_sp{next(_savepoint_names)}"
    conn.execute(f"SAVEPOINT {name}")
    try:
        yield conn
    except BaseException:
        # ROLLBACK TO undoes the block's writes but leaves the savepoint on the
        # stack, so it is released too — otherwise an enclosing block would
        # commit with a stale entry beneath it.
        conn.execute(f"ROLLBACK TO {name}")
        conn.execute(f"RELEASE {name}")
        raise
    conn.execute(f"RELEASE {name}")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS venues (
    id               TEXT PRIMARY KEY,
    name             TEXT NOT NULL,
    address          TEXT,
    latitude         REAL,
    longitude        REAL,
    category         TEXT,
    discovery_source TEXT,
    discovered_at    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS venue_handles (
    venue_id TEXT NOT NULL REFERENCES venues(id) ON DELETE CASCADE,
    handle   TEXT NOT NULL,
    PRIMARY KEY (venue_id, handle)
);

CREATE TABLE IF NOT EXISTS candidate_entities (
    id                 TEXT PRIMARY KEY,
    handle             TEXT NOT NULL UNIQUE,
    state              TEXT NOT NULL DEFAULT 'probationary',
    depth              INTEGER NOT NULL DEFAULT 0,
    mention_count      INTEGER NOT NULL DEFAULT 0,
    mention_sources    TEXT,
    llm_classification TEXT,
    discovery_context  TEXT,
    promoted_venue_id  TEXT REFERENCES venues(id),
    created_at         TEXT NOT NULL,
    updated_at         TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS event_candidates (
    id               TEXT PRIMARY KEY,
    source           TEXT NOT NULL,
    source_type      TEXT NOT NULL,
    url              TEXT,
    image_url        TEXT,
    raw_published_at TEXT,
    title            TEXT,
    description      TEXT,
    venue            TEXT,
    location         TEXT,
    start_time       TEXT,
    end_time         TEXT,
    discovered_at    TEXT NOT NULL,
    raw_data         TEXT
);

CREATE TABLE IF NOT EXISTS weather_cache (
    id         TEXT PRIMARY KEY,
    date       TEXT NOT NULL,
    latitude   TEXT NOT NULL,
    longitude  TEXT NOT NULL,
    data       TEXT NOT NULL,
    fetched_at TEXT NOT NULL,
    UNIQUE (date, latitude, longitude)
);

CREATE TABLE IF NOT EXISTS events (
    id                    TEXT PRIMARY KEY,
    source_type           TEXT NOT NULL,
    url                   TEXT,
    image_url             TEXT,
    title                 TEXT,
    venue                 TEXT,
    venue_id              TEXT REFERENCES venues(id),
    description           TEXT,
    location              TEXT,
    start_time            TEXT,
    end_time              TEXT,
    summary               TEXT,
    summary_embedding     BLOB,
    setting               TEXT NOT NULL DEFAULT 'unknown'
                          CHECK (setting IN ('indoor','outdoor','unknown')),
    timing                TEXT NOT NULL DEFAULT 'exact'
                          CHECK (timing IN ('exact','all_day','unknown')),
    weather               TEXT,
    weather_cache_id      TEXT REFERENCES weather_cache(id),
    astronomical_data     TEXT,
    metadata              TEXT,
    extraction_input_hash TEXT,
    embedding_input_hash  TEXT,
    created_at            TEXT NOT NULL,
    updated_at            TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS event_tags (
    event_id TEXT NOT NULL REFERENCES events(id) ON DELETE CASCADE,
    position INTEGER NOT NULL,
    tag      TEXT NOT NULL,
    weight   REAL NOT NULL,
    PRIMARY KEY (event_id, position)
);

CREATE TABLE IF NOT EXISTS tag_embeddings (
    tag        TEXT NOT NULL,
    model      TEXT NOT NULL,
    embedding  BLOB NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (tag, model)
);

CREATE TABLE IF NOT EXISTS event_source_candidates (
    event_id     TEXT NOT NULL REFERENCES events(id) ON DELETE CASCADE,
    candidate_id TEXT NOT NULL,
    PRIMARY KEY (event_id, candidate_id)
);

CREATE TABLE IF NOT EXISTS event_images (
    event_id TEXT PRIMARY KEY REFERENCES events(id) ON DELETE CASCADE,
    bytes    BLOB NOT NULL
);

CREATE TABLE IF NOT EXISTS preference_embeddings_cache (
    id TEXT PRIMARY KEY,
    file_name TEXT NOT NULL,
    line_hash TEXT NOT NULL,
    line_text TEXT NOT NULL,
    domain TEXT NOT NULL,
    preference_type TEXT NOT NULL,
    embedding BLOB NOT NULL,
    generated_at TEXT NOT NULL,
    UNIQUE(file_name, line_hash)
);

CREATE TABLE IF NOT EXISTS preference_revisions (
    id           TEXT PRIMARY KEY,
    captured_at  TEXT NOT NULL,
    content_hash TEXT NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS preference_embeddings (
    line_hash TEXT NOT NULL,
    model     TEXT NOT NULL,
    embedding BLOB NOT NULL,
    PRIMARY KEY (line_hash, model)
);

CREATE TABLE IF NOT EXISTS preference_lines (
    revision_id     TEXT NOT NULL REFERENCES preference_revisions(id) ON DELETE CASCADE,
    file_name       TEXT NOT NULL,
    position        INTEGER NOT NULL,
    domain          TEXT NOT NULL,
    preference_type TEXT NOT NULL CHECK (preference_type IN ('like','dislike')),
    line_text       TEXT NOT NULL,
    line_hash       TEXT NOT NULL,
    PRIMARY KEY (revision_id, file_name, position)
);

CREATE TABLE IF NOT EXISTS http_cache (
    url           TEXT PRIMARY KEY,
    etag          TEXT,
    last_modified TEXT,
    body          TEXT NOT NULL,
    fetched_at    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS run_history (
    id                     TEXT PRIMARY KEY,
    started_at             TEXT NOT NULL,
    completed_at           TEXT,
    duration_ms            INTEGER,
    steps_completed        TEXT,
    errors                 TEXT,
    skipped_sources        TEXT,
    outcome                TEXT,
    preference_revision_id TEXT REFERENCES preference_revisions(id),
    scoring_config         TEXT
);

CREATE TABLE IF NOT EXISTS feedback (
    id           TEXT PRIMARY KEY,
    event_id     TEXT NOT NULL REFERENCES events(id),
    rating       TEXT NOT NULL,
    submitted_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS event_scores (
    event_id      TEXT NOT NULL REFERENCES events(id) ON DELETE CASCADE,
    run_date      TEXT NOT NULL,
    tag_score     REAL,
    summary_score REAL,
    base_score    REAL NOT NULL,
    match         TEXT NOT NULL,
    PRIMARY KEY (event_id, run_date)
);

CREATE TABLE IF NOT EXISTS score_reasons (
    event_id           TEXT NOT NULL,
    run_date           TEXT NOT NULL,
    position           INTEGER NOT NULL,
    factor             TEXT NOT NULL,
    tag                TEXT,
    matched_preference TEXT,
    similarity         REAL NOT NULL,
    contribution       REAL NOT NULL,
    direction          TEXT NOT NULL,
    PRIMARY KEY (event_id, run_date, position),
    FOREIGN KEY (event_id, run_date)
        REFERENCES event_scores(event_id, run_date) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS recommendations (
    event_id           TEXT NOT NULL REFERENCES events(id) ON DELETE CASCADE,
    run_date           TEXT NOT NULL,
    weather_adjustment REAL NOT NULL,
    tag_confidence     REAL NOT NULL,
    final_score        REAL NOT NULL,
    rank               INTEGER NOT NULL,
    PRIMARY KEY (event_id, run_date),
    FOREIGN KEY (event_id, run_date)
        REFERENCES event_scores(event_id, run_date) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_events_start_time        ON events (start_time);
CREATE INDEX IF NOT EXISTS idx_events_venue_id          ON events (venue_id);
CREATE INDEX IF NOT EXISTS idx_event_tags_tag           ON event_tags (tag);
CREATE INDEX IF NOT EXISTS idx_recommendations_run_rank ON recommendations (run_date, rank);
CREATE INDEX IF NOT EXISTS idx_event_scores_run_date    ON event_scores (run_date);
CREATE INDEX IF NOT EXISTS idx_candidates_source        ON event_candidates (source);
CREATE INDEX IF NOT EXISTS idx_candidates_start_time    ON event_candidates (start_time);
"""


def init_db(db_path: Path | str | None = None) -> None:
    """Initialise the SQLite database and create all tables.

    Idempotent — safe to call multiple times. Uses CREATE TABLE IF NOT EXISTS
    so existing data is never touched.

    Args:
        db_path: Path to the SQLite file. Defaults to database/event_hub.db.
    """
    path = Path(db_path) if db_path is not None else DEFAULT_DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)

    conn = connect(path)
    try:
        # WAL is a property of the file, so it is set here rather than per
        # connection. It is what lets the CLI read while a batch writes.
        conn.execute("PRAGMA journal_mode = WAL")
        conn.executescript(_SCHEMA)
        conn.commit()
    finally:
        conn.close()


def has_schema(db_path: Path | str) -> bool:
    """Report whether a database exists and has been initialised.

    Existence alone proves nothing: `sqlite3.connect` creates a zero-byte file
    for any path it is handed, so a reader that only checked for the file would
    still hit "no such table" on one that a stray connection had created.

    Args:
        db_path: Path to the SQLite file.

    Returns:
        True if the file exists and carries the events and recommendations
        tables, meaning a batch has initialised it.
    """
    path = Path(db_path)
    if not path.exists():
        return False

    conn = sqlite3.connect(path)
    try:
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name IN ('events', 'recommendations')"
        ).fetchall()
    except sqlite3.DatabaseError:
        return False
    finally:
        conn.close()

    return len(rows) == 2
