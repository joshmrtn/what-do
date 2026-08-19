"""A consistent copy of the database, taken before something irreversible.

The churn latch re-keys thousands of rows unattended, at 02:00, with nobody
watching. It is verified inside its transaction and rolls back on failure, but a
rollback only covers what the operation anticipated. A snapshot is the point to
return to when it does not.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from src.storage.sqlite.connection import connect


def snapshot_database(db_path: Path | str, *, reason: str, at: datetime) -> Path:
    """Write a consistent copy beside the database, and return its path.

    `VACUUM INTO`, never a file copy: with WAL on, recent commits live in a
    sidecar, so copying the main file alone captures a torn state. It is also
    safe against a live writer, and produces one file with no sidecars of its
    own.

    Lands beside the database rather than in a configured directory, so it is
    found by whoever finds the database and needs no key that could go unset.

    Args:
        db_path: The database to copy.
        reason: Short slug naming why, e.g. `latch-northshorenightout`. Goes in
            the filename, because a directory of timestamps says nothing.
        at: Injected clock.

    Returns:
        Path to the snapshot.
    """
    source = Path(db_path)
    destination = source.with_name(
        f"{source.stem}-{at.strftime('%Y%m%d-%H%M%S')}-{reason}.db"
    )

    # A name that already exists would make VACUUM INTO fail outright, which is
    # the right default — but two latches in one second is a silly way to lose a
    # run, so the second gets a suffix.
    attempt = 1
    while destination.exists():
        attempt += 1
        destination = source.with_name(
            f"{source.stem}-{at.strftime('%Y%m%d-%H%M%S')}-{reason}-{attempt}.db"
        )

    with connect(source) as conn:
        conn.execute("VACUUM INTO ?", (str(destination),))

    return destination
