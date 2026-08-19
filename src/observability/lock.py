"""Is a batch process alive, asked of the lock the nightly wrapper holds.

The database cannot answer this and the heartbeat cannot either: a killed
process writes nothing on its way out. The lock is the one signal that dies
with the process, which is what makes the difference between *a batch is
working on this* and *a batch died on this*.
"""

from __future__ import annotations

import fcntl
import os
from pathlib import Path

from src.observability.heartbeat import LOCK_PATH


def batch_lock_held(path: Path = LOCK_PATH) -> bool:
    """Whether some process is holding the batch lock right now.

    `flock` offers no read-only interrogation, so the only way to find out is
    to attempt the lock and see whether it is refused.

    **Known and accepted:** in the free branch this holds the batch lock for
    the microseconds between taking it and giving it back. A cron start landing
    inside that window would find it held and skip the night, logged to
    `batch-skipped.log`. It cannot disturb a batch that is *already* running —
    that branch never acquires anything — and the batch is deliberately
    scheduled for 02:00, when nobody is at a keyboard running `--status`. The
    contention-free alternative is parsing `/proc/locks` and matching the
    file's `device:inode`, which is Linux-only and about fifteen lines; this is
    a decision rather than an oversight.

    Args:
        path: The lock file. Never created — an absent one means no batch has
            ever run here, and littering /tmp to discover that is worse than
            the answer.

    Returns:
        True if a process holds it, False if it is free or absent.
    """
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return False
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        return True
    else:
        fcntl.flock(fd, fcntl.LOCK_UN)
        return False
    finally:
        os.close(fd)
