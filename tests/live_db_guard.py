"""Refuses any connection to the live database from inside a test run.

A test that reads `database/event_hub.db` is not isolated, and the failure mode
is silence: it passes for as long as production happens to answer the way the
test wants, then fails somewhere unrelated when production moves. Thirty-five
tests did this for weeks and only surfaced when a column was added to `_SCHEMA`
that the live database had not been migrated to yet.

Lives in an importable module rather than in `conftest.py`, matching
`tier_plugin`, so the tests covering it can load the real plugin instead of a
copy of its logic.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Callable

from src.storage.sqlite.connection import DEFAULT_DB_PATH

_MESSAGE = (
    "A test opened the live database at {path}.\n"
    "\n"
    "Tests must never read or write production data — the result then depends "
    "on what last night's batch happened to leave behind, and the read path "
    "can write.\n"
    "\n"
    "This almost always means a collaborator was left on its production "
    "default. Inject a double for it: `run()` takes load_pairs, "
    "load_all_events, db_ready, load_view_settings, check_freshness, rescore, "
    "load_rescore, probe_status, report_crash and read_progress, and every one "
    "of them builds real repositories against DEFAULT_DB_PATH if you do not.\n"
    "\n"
    "Point the test at tmp_path if it genuinely needs a database of its own."
)

#: The unpatched function, kept so a nested session can put it back.
_original_connect: Callable[..., sqlite3.Connection] | None = None


class LiveDatabaseOpened(BaseException):
    """Raised when a test connects to the real database.

    Deliberately a `BaseException` rather than an `Exception`. Production code
    catches `Exception` broadly and for good reasons — `_default_crash_report`
    swallows everything so a bookkeeping write cannot break a listing — and a
    guard that a legitimate catch can swallow reports green having caught
    nothing, which is the whole failure it exists to prevent.
    """


def _is_the_live_database(target: Any) -> bool:
    """Whether a `sqlite3.connect` argument names the live database.

    Resolved rather than compared as text: a test may reach it as a relative
    path, an absolute one, or through a symlink, and all three are the same
    file. Never raises on an odd argument — `connect` also accepts URIs and
    `:memory:`, and a guard that dies on an input it did not anticipate would
    break runs it has no opinion about.
    """
    try:
        return Path(str(target)).resolve() == DEFAULT_DB_PATH.resolve()
    except (OSError, ValueError):
        return False


def pytest_configure(config: Any) -> None:
    """Wrap `sqlite3.connect` for the session.

    Patched at the library rather than at `storage.db.connect`, because the
    point is that *no* route reaches production — including a raw
    `sqlite3.connect`, which the footgun table already warns against for
    unrelated reasons.
    """
    global _original_connect
    if _original_connect is not None:
        return
    _original_connect = sqlite3.connect

    def guarded(target: Any = None, *args: Any, **kwargs: Any) -> sqlite3.Connection:
        if _is_the_live_database(target):
            raise LiveDatabaseOpened(
                _MESSAGE.format(path=DEFAULT_DB_PATH.resolve())
            )
        return _original_connect(target, *args, **kwargs)

    setattr(sqlite3, "connect", guarded)


def pytest_unconfigure(config: Any) -> None:
    """Put `sqlite3.connect` back, so a nested session leaves nothing behind."""
    global _original_connect
    if _original_connect is not None:
        setattr(sqlite3, "connect", _original_connect)
        _original_connect = None
