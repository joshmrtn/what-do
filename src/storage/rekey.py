"""Move a source onto content-derived candidate ids, in place.

Run twice over the same code: once by hand against the live database, and
thereafter by the churn latch, unattended. Both need the same guarantees, so
there is no migration-script version of this logic — a script that restated the
rule would put rows onto keys the adapters will never generate again.

**Why it cannot be skipped when identity changes.** `reconcile` matches a stored
event to a fresh one by shared candidate id and by nothing else. Change the
scheme without moving the stored rows and every fresh event matches nothing,
becomes a new event, and the stored one is never marked stale — it lingers for
ever. Duplicate candidates are cheap and invisible; duplicate *events* are what
the CLI ranks and shows.

**Everything is verified before the commit, never after.** The `event_scores`
rebuild shipped a broken database on its first attempt by checking afterwards,
by which time the damage was written.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from src.ingestion.candidate_id import derive_content_id
from src.storage.sqlite.connection import connect


@dataclass(frozen=True)
class RekeyOutcome:
    """What one re-key did, for a run summary or a migration's assertions."""

    source: str
    candidates_before: int
    candidates_after: int
    #: Duplicate rows for a listing already represented. Deleted, not merged:
    #: what they published survives in `candidate_versions`.
    absorbed: int
    versions_absorbed: int
    #: Pairs that became a comparison of one candidate against itself. Not a
    #: judgement about anything, and 77% of the live corpus.
    self_pairs_removed: int
    decisions_rekeyed: int
    #: Collapsed ids now claimed by more than one event. Left alone — reconcile
    #: merges them, and it skips superseded rows when it does — but reported,
    #: because it should be rare and a migration will want to assert on it.
    candidates_shared_by_several_events: int


class RekeyFailed(RuntimeError):
    """A verification inside the transaction failed, and nothing was written."""


def rekey_to_content_ids(db_path: Path | str, *, source: str) -> RekeyOutcome:
    """Re-key one source's stored candidates onto `derive_content_id`.

    Args:
        db_path: Path to the SQLite database.
        source: The feed to move. Keyed on `source`, never `source_type`, which
            is a category covering feeds with opposite identity behaviour.

    Returns:
        What the operation did.

    Raises:
        RekeyFailed: If a verification failed. The transaction is rolled back,
            so the database is untouched.
    """
    conn = connect(db_path)
    # Manual transaction control. `defer_foreign_keys` is switched off again at
    # every COMMIT or ROLLBACK, so it has to be set *inside* the transaction it
    # applies to — and the driver's implicit BEGIN would otherwise open that
    # transaction on the first statement, after the pragma had already lapsed.
    conn.isolation_level = None
    try:
        conn.execute("BEGIN IMMEDIATE")
        # The parent key of `candidate_versions.candidate_id` is about to move
        # and there is no ON UPDATE CASCADE, so immediate enforcement would
        # reject the first delete. Deferring holds the check to the commit,
        # where `foreign_key_check` below still has to pass first.
        conn.execute("PRAGMA defer_foreign_keys = ON")
        # Read before anything moves: the invariant is that no event *loses*
        # its last candidate, which cannot be checked against the end state
        # alone. Synthetic activities enter the pipeline as pre-structured
        # events and never had one — six of them live — so a global check
        # would refuse a re-key that is doing no harm.
        claimants = _events_holding_candidates(conn)
        outcome = _rekey(conn, source)
        _verify(conn, source, outcome, claimants)
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise
    finally:
        conn.close()

    return outcome


def _rekey(conn: sqlite3.Connection, source: str) -> RekeyOutcome:
    rows = conn.execute(
        "SELECT id, title, venue, start_time, discovered_at, last_seen_at "
        "FROM event_candidates WHERE source = ?",
        (source,),
    ).fetchall()

    mapping: dict[str, str] = {}
    for old_id, title, venue, start_time, _, _ in rows:
        mapping[old_id] = derive_content_id(
            source=source,
            title=title,
            venue=venue,
            start=_parse(start_time),
        )

    survivors = _collapse_candidates(conn, rows, mapping)
    versions_absorbed = _collapse_versions(conn, mapping)
    self_pairs, rekeyed = _rewrite_decisions(conn, mapping)
    _rewrite_links(conn, mapping)

    return RekeyOutcome(
        source=source,
        candidates_before=len(rows),
        candidates_after=len(survivors),
        absorbed=len(rows) - len(survivors),
        versions_absorbed=versions_absorbed,
        self_pairs_removed=self_pairs,
        decisions_rekeyed=rekeyed,
        candidates_shared_by_several_events=_shared_candidates(conn, survivors),
    )


def _collapse_candidates(
    conn: sqlite3.Connection,
    rows: list[tuple[str, str | None, str | None, str | None, str, str]],
    mapping: dict[str, str],
) -> set[str]:
    """Keep one row per content id, with the group's true sighting window.

    The survivor takes the **earliest** `discovered_at` and the **latest**
    `last_seen_at` in its group, which is the first moment either column has
    told the truth for a feed that re-mints: every row currently reads
    `last_seen_at == discovered_at`, because no id was ever seen twice.

    The row kept is the **most recently seen** one, because its other columns
    are what the listing currently says — the earlier members are the same
    listing as it used to read, and until the next fetch overwrites the row it
    is what the pipeline works from. Ties break on the id so a group seen
    entirely on one night still resolves the same way every time.
    """
    groups: dict[str, list[tuple[str, str, str]]] = {}
    for old_id, _, _, _, discovered_at, last_seen_at in rows:
        groups.setdefault(mapping[old_id], []).append(
            (old_id, discovered_at, last_seen_at)
        )

    for new_id, members in groups.items():
        keeper = max(members, key=lambda m: (m[2], m[0]))[0]
        earliest = min(m[1] for m in members)
        latest = max(m[2] for m in members)

        # Delete the absorbed rows first: the keeper is about to take an id one
        # of them may already hold, and the primary key would collide.
        conn.executemany(
            "DELETE FROM event_candidates WHERE id = ?",
            [(m[0],) for m in members if m[0] != keeper],
        )
        conn.execute(
            "UPDATE event_candidates SET id = ?, discovered_at = ?, last_seen_at = ? "
            "WHERE id = ?",
            (new_id, earliest, latest, keeper),
        )

    return set(groups)


def _collapse_versions(conn: sqlite3.Connection, mapping: dict[str, str]) -> int:
    """Re-key the version history, keeping the earliest sighting of each content.

    Rows sharing a content hash are the same publication seen again, so the
    earliest `observed_at` is the one that says when it first appeared. Distinct
    hashes are genuine upstream edits and all survive — which is the point: this
    feed has never been able to record one.
    """
    rows = conn.execute(
        "SELECT candidate_id, content_hash, observed_at, payload "
        "FROM candidate_versions"
    ).fetchall()
    touched = [row for row in rows if row[0] in mapping]
    if not touched:
        return 0

    keep: dict[tuple[str, str], tuple[str, str]] = {}
    for candidate_id, content_hash, observed_at, payload in touched:
        key = (mapping[candidate_id], content_hash)
        if key not in keep or observed_at < keep[key][0]:
            keep[key] = (observed_at, payload)

    conn.executemany(
        "DELETE FROM candidate_versions WHERE candidate_id = ?",
        [(old_id,) for old_id in mapping],
    )
    conn.executemany(
        "INSERT INTO candidate_versions (candidate_id, content_hash, observed_at, "
        "payload) VALUES (?, ?, ?, ?)",
        [(cid, chash, seen, load) for (cid, chash), (seen, load) in keep.items()],
    )

    return len(touched) - len(keep)


def _rewrite_links(conn: sqlite3.Connection, mapping: dict[str, str]) -> None:
    """Point every event at the surviving candidate, dropping duplicate links.

    An event claiming three rows of one listing ends with one link, not none —
    losing every link would make the event unreachable to `reconcile` and it
    would be minted afresh, which is the duplicate-event failure this whole
    operation exists to avoid.
    """
    rows = conn.execute(
        "SELECT event_id, candidate_id FROM event_source_candidates"
    ).fetchall()
    touched = [row for row in rows if row[1] in mapping]
    if not touched:
        return

    conn.executemany(
        "DELETE FROM event_source_candidates WHERE candidate_id = ?",
        [(old_id,) for old_id in mapping],
    )
    conn.executemany(
        "INSERT OR IGNORE INTO event_source_candidates (event_id, candidate_id) "
        "VALUES (?, ?)",
        sorted({(event_id, mapping[cid]) for event_id, cid in touched}),
    )


def _rewrite_decisions(
    conn: sqlite3.Connection, mapping: dict[str, str]
) -> tuple[int, int]:
    """Re-key the dedup corpus, discarding pairs that became self-comparisons.

    A pair whose sides collapse onto one id is not a judgement about anything —
    it recorded that two rows of the same listing looked alike, which was always
    true and is the artefact the re-key removes. Live that is 2192 of 2843 rows,
    and a model trained on them learns that identical strings match.

    Pair order is normalised because the primary key is `(pass, a, b)`: two
    re-keyed pairs can land on one identity in either order.
    """
    rows = conn.execute(
        "SELECT pass_name, record_kind, record_a, record_b, score, verdict, "
        "stratum, sample_denominator, content_hash_a, content_hash_b, run_id, "
        "updated_at FROM dedup_decisions"
    ).fetchall()

    self_pairs = 0
    rewritten: dict[tuple[str, str, str], tuple[object, ...]] = {}
    stale: list[tuple[str, str, str]] = []

    for row in rows:
        pass_name, kind, a, b = row[0], row[1], row[2], row[3]
        if a not in mapping and b not in mapping:
            continue

        stale.append((pass_name, a, b))
        new_a, new_b = mapping.get(a, a), mapping.get(b, b)
        if new_a == new_b:
            self_pairs += 1
            continue

        if new_b < new_a:
            new_a, new_b = new_b, new_a
            row = (pass_name, kind, new_a, new_b, *row[4:8], row[9], row[8], *row[10:])
        else:
            row = (pass_name, kind, new_a, new_b, *row[4:])
        rewritten[(pass_name, new_a, new_b)] = row

    conn.executemany(
        "DELETE FROM dedup_decisions WHERE pass_name = ? AND record_a = ? "
        "AND record_b = ?",
        stale,
    )
    conn.executemany(
        "INSERT OR REPLACE INTO dedup_decisions (pass_name, record_kind, record_a, "
        "record_b, score, verdict, stratum, sample_denominator, content_hash_a, "
        "content_hash_b, run_id, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, "
        "?, ?)",
        list(rewritten.values()),
    )

    return self_pairs, len(rewritten)


def _shared_candidates(conn: sqlite3.Connection, survivors: set[str]) -> int:
    if not survivors:
        return 0
    placeholders = ",".join("?" * len(survivors))
    return int(
        conn.execute(
            f"SELECT COUNT(*) FROM (SELECT candidate_id FROM event_source_candidates "
            f"WHERE candidate_id IN ({placeholders}) GROUP BY candidate_id "
            f"HAVING COUNT(DISTINCT event_id) > 1)",
            tuple(survivors),
        ).fetchone()[0]
    )


def _events_holding_candidates(conn: sqlite3.Connection) -> set[str]:
    """Events that had at least one candidate before anything moved."""
    return {
        row[0]
        for row in conn.execute("SELECT DISTINCT event_id FROM event_source_candidates")
    }


def _verify(
    conn: sqlite3.Connection,
    source: str,
    outcome: RekeyOutcome,
    claimants: set[str],
) -> None:
    """Everything that must hold, checked while a rollback is still possible."""
    dangling = conn.execute("PRAGMA foreign_key_check").fetchall()
    if dangling:
        raise RekeyFailed(f"{len(dangling)} dangling references after re-key")

    remaining = conn.execute(
        "SELECT id, title, venue, start_time FROM event_candidates WHERE source = ?",
        (source,),
    ).fetchall()
    if len(remaining) != outcome.candidates_after:
        raise RekeyFailed(
            f"expected {outcome.candidates_after} rows for {source}, "
            f"found {len(remaining)}"
        )

    # Every surviving row must be keyed on its *own* content. A row that is not
    # would never be recognised again by the adapter that wrote it.
    for row_id, title, venue, start_time in remaining:
        expected = derive_content_id(
            source=source, title=title, venue=venue, start=_parse(start_time)
        )
        if row_id != expected:
            raise RekeyFailed(f"row {row_id!r} is not keyed on its own content")

    stranded = claimants - _events_holding_candidates(conn)
    if stranded:
        raise RekeyFailed(
            f"{len(stranded)} event(s) lost their last candidate — reconcile "
            "would mint a duplicate for every one of them"
        )


def _parse(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value) if value else None
