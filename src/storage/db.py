"""SQLite database initialisation and schema management."""

from __future__ import annotations

import sqlite3
from pathlib import Path

_DEFAULT_DB_PATH = Path("database/event_hub.db")

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
    mention_count INTEGER NOT NULL DEFAULT 0,
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
    title TEXT,
    venue TEXT,
    description TEXT,
    location TEXT,
    start_time TEXT,
    end_time TEXT,
    tags TEXT,
    summary TEXT,
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
    score REAL NOT NULL,
    tier TEXT NOT NULL,
    match TEXT NOT NULL,
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
    path = Path(db_path) if db_path is not None else _DEFAULT_DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(path)
    try:
        conn.executescript(_SCHEMA)
        conn.commit()
    finally:
        conn.close()
