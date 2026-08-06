"""SQLite database initialisation and schema management."""

from __future__ import annotations

import sqlite3
from pathlib import Path

DEFAULT_DB_PATH = Path("database/event_hub.db")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS venues (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    address TEXT,
    latitude REAL,
    longitude REAL,
    category TEXT,
    social_handles TEXT,
    blocklisted INTEGER NOT NULL DEFAULT 0,
    discovery_source TEXT,
    discovered_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS candidate_entities (
    id TEXT PRIMARY KEY,
    handle TEXT NOT NULL UNIQUE,
    state TEXT NOT NULL DEFAULT 'probationary',
    depth INTEGER NOT NULL DEFAULT 0,
    mention_count INTEGER NOT NULL DEFAULT 0,
    mention_sources TEXT,
    llm_classification TEXT,
    discovery_context TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS event_candidates (
    id TEXT PRIMARY KEY,
    source TEXT NOT NULL,
    source_type TEXT NOT NULL,
    url TEXT,
    image_url TEXT,
    raw_published_at TEXT,
    title TEXT,
    description TEXT,
    venue TEXT,
    location TEXT,
    start_time TEXT,
    end_time TEXT,
    discovered_at TEXT NOT NULL,
    raw_data TEXT
);

CREATE TABLE IF NOT EXISTS events (
    id TEXT PRIMARY KEY,
    source_event_candidates TEXT NOT NULL,
    source_type TEXT NOT NULL,
    url TEXT,
    image_url TEXT,
    image_bytes BLOB,
    title TEXT,
    venue TEXT,
    description TEXT,
    location TEXT,
    start_time TEXT,
    end_time TEXT,
    tags TEXT,
    summary TEXT,
    setting TEXT NOT NULL DEFAULT 'unknown',
    tag_embeddings BLOB,
    summary_embedding BLOB,
    weather TEXT,
    astronomical_data TEXT,
    metadata TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS recommendations (
    id TEXT PRIMARY KEY,
    event_id TEXT NOT NULL,
    run_date TEXT NOT NULL,
    base_score REAL NOT NULL,
    weather_adjustment REAL NOT NULL,
    tag_confidence REAL NOT NULL,
    final_score REAL NOT NULL,
    tier TEXT NOT NULL,
    match TEXT NOT NULL,
    rank INTEGER NOT NULL,
    reasons TEXT NOT NULL,
    FOREIGN KEY (event_id) REFERENCES events(id)
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

CREATE TABLE IF NOT EXISTS weather_cache (
    id TEXT PRIMARY KEY,
    date TEXT NOT NULL,
    latitude REAL NOT NULL,
    longitude REAL NOT NULL,
    data TEXT NOT NULL,
    fetched_at TEXT NOT NULL,
    UNIQUE(date, latitude, longitude)
);

CREATE TABLE IF NOT EXISTS http_cache (
    url TEXT PRIMARY KEY,
    etag TEXT,
    last_modified TEXT,
    body TEXT NOT NULL,
    fetched_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS run_history (
    id TEXT PRIMARY KEY,
    started_at TEXT NOT NULL,
    completed_at TEXT,
    duration_ms INTEGER,
    steps_completed TEXT,
    errors TEXT,
    outcome TEXT
);

CREATE TABLE IF NOT EXISTS feedback (
    id TEXT PRIMARY KEY,
    event_id TEXT NOT NULL,
    rating TEXT NOT NULL,
    submitted_at TEXT NOT NULL,
    FOREIGN KEY (event_id) REFERENCES events(id)
);

CREATE TABLE IF NOT EXISTS blocklist (
    id TEXT PRIMARY KEY,
    value TEXT NOT NULL UNIQUE,
    loaded_at TEXT NOT NULL
);
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

    conn = sqlite3.connect(path)
    try:
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
