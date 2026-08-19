"""Whether a batch is holding the lock, asked without disturbing one."""

from __future__ import annotations

import fcntl
import os

from src.observability.lock import batch_lock_held


def test_an_unheld_lock_reads_as_free(tmp_path):
    path = tmp_path / "batch.lock"
    path.write_text("")

    assert batch_lock_held(path) is False


def test_a_held_lock_reads_as_held(tmp_path):
    """`flock` has no read-only interrogation — the only way to learn whether a
    lock is held is to attempt it. Two open file descriptions conflict even
    within one process, which is what makes this testable at all."""
    path = tmp_path / "batch.lock"
    path.write_text("")
    fd = os.open(path, os.O_RDONLY)
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        assert batch_lock_held(path) is True
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def test_the_probe_gives_the_lock_straight_back(tmp_path):
    """It takes the lock for microseconds in the free branch, and a probe that
    kept it would block the 02:00 batch outright rather than in a window."""
    path = tmp_path / "batch.lock"
    path.write_text("")

    batch_lock_held(path)

    fd = os.open(path, os.O_RDONLY)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def test_a_lock_file_that_has_never_existed_reads_as_free(tmp_path):
    """No file means no batch has ever run on this machine. Creating one to
    find that out would leave litter in /tmp on every `--status`."""
    path = tmp_path / "never-created.lock"

    assert batch_lock_held(path) is False
    assert not path.exists()
