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
    discovered_at    TEXT NOT NULL,
    -- Which spelling is the real one, for venue matching (#11). Adding it does
    -- not fix the dedup bug on its own: dedup compares `events.venue` free text,
    -- so `The Rhumb Line` vs `Rhumb Line` still needs normalising inside
    -- `venues_match`.
    canonical_name   TEXT
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
    -- How much is known about *when*. Missing until 2026-08-11, which is why
    -- every ICS `all_day` was reloaded as `exact`: the candidate round trip
    -- dropped it, and the batch prefers the loaded copy.
    timing           TEXT NOT NULL DEFAULT 'exact',
    -- Authored by adapters for sources that state everything they know. NULL
    -- and empty for every source that leaves both to extraction.
    summary          TEXT,
    tags             TEXT,
    metadata         TEXT,
    -- Last observation, against `discovered_at`'s first. Declared last because
    -- ALTER TABLE ADD COLUMN appends, so a fresh build and the migrated live
    -- database agree on position as well as on name.
    --
    -- Nullable, deliberately: NOT NULL added by ALTER needs a DEFAULT that then
    -- cannot be dropped without a table rebuild, and a fresh database carrying a
    -- default the live one lacks is exactly the drift the schema check exists to
    -- catch. The writer always supplies it and the reader raises on NULL.
    last_seen_at     TEXT
);

-- What a source published, retained when it publishes something else.
--
-- `event_candidates` holds one row per listing and is overwritten by each
-- re-fetch; without this, an edited listing silently destroys the text we
-- ingested, deduped and extracted against, and a stored dedup decision can no
-- longer be checked against what was really published (#27).
--
-- The content hash is half the primary key, which is what makes an unchanged
-- re-fetch an INSERT OR IGNORE no-op: no read, no comparison, and no way for a
-- caller to forget to do one. Rows therefore accrue only for real edits —
-- measured at 2.26% of candidates per re-fetch.
--
-- `payload` is JSON rather than columns so that adding a field to
-- `EventCandidate` cannot leave the history behind: `PUBLISHED_FIELDS` builds
-- the payload and the fingerprint together.
CREATE TABLE IF NOT EXISTS candidate_versions (
    candidate_id TEXT NOT NULL REFERENCES event_candidates(id),
    content_hash TEXT NOT NULL,
    -- When this content was FIRST seen. `event_candidates.last_seen_at` already
    -- answers "are we still seeing it"; a version that moved its own stamp
    -- could not say when the text actually changed.
    observed_at  TEXT NOT NULL,
    payload      TEXT NOT NULL,
    PRIMARY KEY (candidate_id, content_hash)
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

-- Air quality is a separate service from the forecast, with its own shorter
-- horizon, so it gets its own table rather than a discriminator column on
-- weather_cache: the UNIQUE key there is (date, latitude, longitude), which two
-- providers would collide on. Same shape, same reader.
CREATE TABLE IF NOT EXISTS air_quality_cache (
    id         TEXT PRIMARY KEY,
    date       TEXT NOT NULL,
    latitude   TEXT NOT NULL,
    longitude  TEXT NOT NULL,
    data       TEXT NOT NULL,
    fetched_at TEXT NOT NULL,
    UNIQUE (date, latitude, longitude)
);

-- What TMDb said, keyed by the question asked. RAW: it is the provider's
-- answer, retained and expirable. What we conclude a film *is* belongs in
-- movie_metadata, which is a separate table and a separate decision.
--
-- A miss is a row with is_miss, never an absent row. An absent row cannot be
-- told apart from never having asked, which is what makes an unrecognised
-- title a request that repeats on every run for ever.
CREATE TABLE IF NOT EXISTS tmdb_responses (
    id         TEXT PRIMARY KEY,
    title_key  TEXT NOT NULL,
    year       TEXT NOT NULL,
    payload    TEXT NOT NULL,
    is_miss    INTEGER NOT NULL,
    fetched_at TEXT NOT NULL,
    UNIQUE (title_key, year)
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
    updated_at            TEXT NOT NULL,
    -- Everything below was added in one additive pass, and is listed in the
    -- order the live database's `ALTER TABLE ADD COLUMN` statements applied it.
    -- `ALTER` can only append, so keeping this order means a fresh database and
    -- the migrated one agree column for column rather than merely as sets.
    --
    -- Most of these are deliberately inert: the column exists, nothing reads or
    -- writes it, and the feature named below will wire it. A column is cheap to
    -- add now and expensive later, because migrating means a hand operation
    -- against a database holding ~40h of compute.
    --
    -- Which event replaced this one, once dedup soft-deletes instead of
    -- deleting outright. Inert until the integrity-gaps pass.
    superseded_by             TEXT REFERENCES events(id),
    superseded_at             TEXT,
    -- Why dedup merged: `fuzzy` / `semantic` / `reconcile`, and how close the
    -- match was. Today a merge records nothing, so a past merge cannot be
    -- explained at all. No CHECK constraint on `merged_by` — SQLite cannot add
    -- one via ALTER, so constraining it here would make fresh databases
    -- strictly stricter than the migrated one. Code enforces the values.
    merged_by                 TEXT,
    merge_similarity          REAL,
    -- Inert: arrival class, and issue #5's "when was this posted".
    arrival                   TEXT,
    posted_at                 TEXT,
    -- Which model and prompt produced this row's tags. Written from the night
    -- these landed; rows extracted before have neither and never will. Without
    -- them a row fit for the confidence curve is indistinguishable from one
    -- carrying tags from a prompt since fixed.
    extraction_model          TEXT,
    extraction_prompt_version TEXT,
    -- Every way the reply that produced this row's tags fell short, or NULL
    -- where it met the schema in full. A thin answer used to be discarded, so
    -- the row kept older tags and no record that tonight's run had rejected
    -- anything. It is also what keeps the shortest inputs in the confidence
    -- curve's dataset instead of holing it exactly where it is most sensitive.
    extraction_degradation    TEXT,
    -- What the model was actually asked, kept beside what it answered. Only the
    -- hash was stored, so the corpus a refit learns from was current state
    -- rather than an observation. Declared last because ALTER TABLE appends, so
    -- a fresh build and the migrated live database agree on position too.
    extraction_input          TEXT,
    extraction_input_chars    INTEGER,
    -- The feed, as against `source_type`'s category. Declared last because
    -- ALTER TABLE appends.
    source                    TEXT
);

-- The tag-confidence curve currently in force, and how it got there.
--
-- One row. The refit runs at the end of a batch and writes here; the *next*
-- run reads it, so a night is scored with constants that did not move under it.
-- Absent or empty means the config defaults stand, which is the state of a
-- fresh deployment and of any regime that has not armed.
-- Every extraction, kept. `events` holds the latest one and re-extraction
-- overwrites it, so a corpus read from there is current state rather than a
-- record of what the model was asked and answered — and the two observations
-- most worth having, either side of a prompt change, are exactly the ones
-- overwriting destroys.
--
-- Append-only. `observed_at` is when the extraction ran, which is the only
-- honest chronology: `events.created_at` is when the *event* was created, and
-- sorting a series by it reads re-extracted rows in an order unrelated to when
-- their tags were produced.
CREATE TABLE IF NOT EXISTS extraction_observations (
    event_id          TEXT NOT NULL REFERENCES events(id),
    observed_at       TEXT NOT NULL,
    chars             INTEGER NOT NULL,
    tags              INTEGER NOT NULL,
    model             TEXT,
    prompt_version    TEXT,
    degradation       TEXT,
    source            TEXT,
    -- True where the row was reconstructed from an event rather than recorded
    -- as it happened, so a later reader can tell evidence from inference.
    backfilled        INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (event_id, observed_at)
);

CREATE TABLE IF NOT EXISTS curve_state (
    id                INTEGER PRIMARY KEY CHECK (id = 1),
    cap               REAL NOT NULL,
    saturation        REAL NOT NULL,
    regime            TEXT,
    updated_at        TEXT NOT NULL,
    -- The decision behind it, as JSON: row counts, both held-out scores, the
    -- per-source multipliers, any change points. A stored score is only
    -- interpretable against the fit that produced it.
    provenance        TEXT
);

-- Accumulated evidence about whether a source's publisher can identify its own
-- listings, and the one-way latch that acts on it. Derived: recomputable from
-- `event_candidates` history, which is why it lives here and the human's policy
-- lives in config.
CREATE TABLE IF NOT EXISTS source_identity_state (
    source           TEXT PRIMARY KEY,
    -- Churned listings summed over every qualifying run, never reset. A streak
    -- is reset by a quiet night, and a feed churning every other night would
    -- never reach two consecutive runs while duplicating for ever.
    churn_evidence   INTEGER NOT NULL DEFAULT 0,
    qualifying_runs  INTEGER NOT NULL DEFAULT 0,
    -- When the latch fired. One-way: once a publisher's ids have been shown to
    -- identify nothing, they are never trusted again. Content-keyed ids read 0%
    -- churn by construction, so anything re-evaluating both ways would
    -- oscillate, re-keying and minting duplicates each time.
    latched_at       TEXT,
    updated_at       TEXT NOT NULL
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
    scoring_config         TEXT,
    -- The dedup thresholds in force, for the same reason `scoring_config`
    -- records the scoring ones: a verdict is a function of numbers that will
    -- be tuned, and a retuned threshold otherwise reinterprets every label
    -- already stored. Wired by the dedup provenance work.
    dedup_config           TEXT
);

-- Every comparison the dedup passes made, not only the ones that merged.
--
-- A surviving row can say "I was merged into X"; it cannot say "I was compared
-- against X and judged different", and that is most of what a dedup model
-- learns from. Measured on the live corpus: two merges against 1,629
-- considered-and-rejected pairs.
--
-- Derived, in the raw/derived split: regenerable by recomputing the guards and
-- the scores over retained records. What is *not* regenerable is the verdict
-- under the thresholds in force at the time, which is why `run_id` is here.
--
-- No foreign key on `record_a`/`record_b`: they are polymorphic across
-- `record_kind` (a candidate id for Pass 1, an event id for Pass 2) and SQLite
-- cannot express that. So `check_references` cannot see this table — the one
-- place in the schema where a dangling reference would not be reported.
CREATE TABLE IF NOT EXISTS dedup_decisions (
    pass_name          TEXT    NOT NULL,   -- fuzzy | semantic | reconcile
    record_kind        TEXT    NOT NULL,   -- candidate | event
    record_a           TEXT    NOT NULL,   -- the lexicographically smaller id
    record_b           TEXT    NOT NULL,
    score              REAL    NOT NULL,
    verdict            TEXT    NOT NULL,   -- merged | distinct
    stratum            TEXT    NOT NULL,   -- merged | near_miss | sampled
    sample_denominator INTEGER NOT NULL,   -- 1 where the stratum was kept whole
    content_hash_a     TEXT    NOT NULL,   -- of the text actually compared
    content_hash_b     TEXT    NOT NULL,
    run_id             TEXT    NOT NULL REFERENCES run_history(id),
    updated_at         TEXT    NOT NULL,
    PRIMARY KEY (pass_name, record_a, record_b)
);

CREATE TABLE IF NOT EXISTS feedback (
    id           TEXT PRIMARY KEY,
    event_id     TEXT NOT NULL REFERENCES events(id),
    rating       TEXT NOT NULL,
    submitted_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS event_scores (
    event_id       TEXT NOT NULL REFERENCES events(id) ON DELETE CASCADE,
    run_date       TEXT NOT NULL,
    tag_score      REAL,
    summary_score  REAL,
    base_score     REAL NOT NULL,
    -- A pure function of the event's own tag count, so it belongs with the
    -- verdict rather than with the placement. It sat on `recommendations`
    -- until 2026-08-11, which is what made the score/ranking split look muddy.
    tag_confidence REAL NOT NULL DEFAULT 1.0,
    match          TEXT NOT NULL,
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

-- Only in-scope events get a row. An event that was scored but fell outside
-- the night's window has an `event_scores` row and no ranking, which is what
-- lets a score outlive the run that declined to rank it.
CREATE TABLE IF NOT EXISTS rankings (
    event_id           TEXT NOT NULL REFERENCES events(id) ON DELETE CASCADE,
    run_date           TEXT NOT NULL,
    weather_adjustment REAL NOT NULL,
    final_score        REAL NOT NULL,
    rank               INTEGER NOT NULL,
    PRIMARY KEY (event_id, run_date),
    FOREIGN KEY (event_id, run_date)
        REFERENCES event_scores(event_id, run_date) ON DELETE CASCADE
);

-- One row per read-time rescore, never an update of the last.
--
-- After a rescore, a run date holds numbers produced by a different forecast
-- from the one its own row describes, and nothing else records that. A
-- `rescored_at` column on `run_history` would answer "when last?" and destroy
-- the answer to "how did this move, and how often?" — the same reasoning that
-- made dedup provenance a decision log rather than a column on the survivor.
CREATE TABLE IF NOT EXISTS rescores (
    id                     TEXT PRIMARY KEY,
    run_date               TEXT NOT NULL,
    rescored_at            TEXT NOT NULL,
    -- When the forecast it scored against was issued. NULL when no event in
    -- the run carried one, which is an all-indoor listing rather than a
    -- failure.
    forecast_issued_at     TEXT,
    preference_revision_id TEXT REFERENCES preference_revisions(id),
    events_rescored        INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_rescores_run_date         ON rescores (run_date, rescored_at);
CREATE INDEX IF NOT EXISTS idx_events_start_time        ON events (start_time);
CREATE INDEX IF NOT EXISTS idx_events_venue_id          ON events (venue_id);
CREATE INDEX IF NOT EXISTS idx_event_tags_tag           ON event_tags (tag);
CREATE INDEX IF NOT EXISTS idx_rankings_run_rank        ON rankings (run_date, rank);
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
        True if the file exists and carries the events and rankings
        tables, meaning a batch has initialised it.
    """
    path = Path(db_path)
    if not path.exists():
        return False

    conn = sqlite3.connect(path)
    try:
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name IN ('events', 'rankings')"
        ).fetchall()
    except sqlite3.DatabaseError:
        return False
    finally:
        conn.close()

    return len(rows) == 2
