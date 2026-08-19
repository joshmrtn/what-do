"""The batch lock is one literal in two files that cannot import each other.

`scripts/run-batch.sh` takes the lock; `what-do --status` reads it. A wrapper
that asked Python for the path at startup would remove the duplication, at the
cost of editing a script while it is executing and of moving a lock a running
batch may hold — so the literal stays in both places and this fails if they
ever stop agreeing.
"""

from __future__ import annotations

from pathlib import Path

from src.observability.heartbeat import LOCK_PATH

_WRAPPER = Path(__file__).resolve().parents[2] / "scripts" / "run-batch.sh"


def test_the_wrapper_locks_the_path_status_reads():
    """Not a file-existence assertion in the footgun's sense — it fails on
    *disagreement*, and a wrapper that has gone missing failing loudly here is
    the correct outcome rather than an accident."""
    assert f'LOCK_FILE="{LOCK_PATH}"' in _WRAPPER.read_text()
