"""Live batch state, written beside the lock and read by `what-do --status`.

Deliberately not a database row. A heartbeat is neither raw nor derived: it
records nothing anyone will query tomorrow, it expires the moment the process
does, and it exists to answer one question — *what is this run doing right
now* — that the database cannot answer at all, because a killed process writes
no row saying it was killed.

Read together with the lock, the pair distinguishes four states where
`run_history` alone can only manage two. See `presentation/status.py`.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from src.observability.reporter import FINISHED, Progress

#: Where the nightly wrapper puts its lock. The literal also lives in
#: `scripts/run-batch.sh`, which cannot import Python, and
#: `tests/unit/test_lock_path_is_shared.py` fails if the two ever disagree.
#: Deliberately not unified by having the wrapper ask Python at startup: that
#: edits a script while it is executing to remove a duplication a test already
#: pins for free.
LOCK_PATH = Path("/tmp/what-do-batch.lock")

#: Beside the lock, and for the same reason — this is state about a process,
#: not about the data. A reboot clears both, which is correct: after one,
#: neither the lock nor the run it described exists.
HEARTBEAT_PATH = Path("/tmp/what-do-batch-progress.json")


@dataclass(frozen=True)
class Item:
    """One event, as the heartbeat names it.

    `since` is when this became true: when the item went to the model for
    `in_flight`, when it came back for `last_completed`.
    """

    item_id: str
    label: str
    since: datetime


@dataclass(frozen=True)
class Heartbeat:
    """What a run was doing when it last said anything."""

    run_id: str
    stage: str
    done: int
    total: int
    started_at: datetime
    updated_at: datetime
    in_flight: Item | None = None
    last_completed: Item | None = None
    deadline: datetime | None = None


class HeartbeatFile:
    """Writes the live state on every report, and removes it on a clean exit.

    Every report, not every milestone: the throttle in `ProgressLog` exists to
    ration a *log a person reads*, and this is a single small file being
    replaced. At a hosted provider's speed that is a few thousand writes an
    hour, against the ~54,000 row writes checkpointing alone used to cost in a
    night.
    """

    def __init__(self, path: Path, *, run_id: str) -> None:
        self._path = path
        self._run_id = run_id
        self._stage: str | None = None
        self._started_at: datetime | None = None
        self._last_completed: Item | None = None

    def __call__(self, report: Progress) -> None:
        """Record one report. Never raises — see `_write`."""
        if self._stage != report.stage:
            # A new stage is a new reckoning. One file follows the run through
            # every stage of it, so carrying extraction's start into embedding
            # would report a rate computed over the hours before embedding
            # began — and the last event extraction finished as the last thing
            # embedding did.
            self._stage = report.stage
            self._started_at = report.now
            self._last_completed = None
        started_at = self._started_at if self._started_at is not None else report.now
        item = Item(item_id=report.item_id, label=report.label, since=report.now)
        finished = report.phase == FINISHED
        if finished:
            self._last_completed = item
        self._write(
            {
                "run_id": self._run_id,
                "stage": report.stage,
                "done": report.done,
                "total": report.total,
                "started_at": started_at.isoformat(),
                "updated_at": report.now.isoformat(),
                "in_flight": None if finished else _item(item),
                "last_completed": _item(self._last_completed),
                "deadline": (
                    report.deadline.isoformat() if report.deadline is not None else None
                ),
            }
        )

    def clear(self) -> None:
        """Remove the file, because this run is over and said so.

        A process that is killed cannot reach here, which is the point: what is
        left behind is the evidence.
        """
        try:
            self._path.unlink(missing_ok=True)
        except OSError:
            pass

    def _write(self, document: dict[str, object]) -> None:
        """Replace the file atomically, and never fail the run for trying.

        Temp file in the same directory then `os.replace`, so a reader either
        sees the previous document or the next one and never half of either —
        `--status` reads this while it is being written.
        """
        try:
            with tempfile.NamedTemporaryFile(
                "w", dir=self._path.parent, prefix=".progress-", delete=False
            ) as handle:
                json.dump(document, handle)
                temp = Path(handle.name)
            os.replace(temp, self._path)
        except OSError:
            # Reporting is a courtesy to whoever is watching. A full disk must
            # not cost eight hours of model time.
            return


def _item(item: Item | None) -> dict[str, str] | None:
    if item is None:
        return None
    return {"id": item.item_id, "label": item.label, "since": item.since.isoformat()}


def _read_item(raw: object) -> Item | None:
    if not isinstance(raw, dict):
        return None
    return Item(
        item_id=str(raw["id"]),
        label=str(raw["label"]),
        since=datetime.fromisoformat(str(raw["since"])),
    )


def read_heartbeat(path: Path = HEARTBEAT_PATH) -> Heartbeat | None:
    """What the running batch last said, or None if it said nothing readable.

    None covers three cases on purpose — no file, an unparseable one, and one
    written by a version that recorded different fields. All three mean the
    same thing to a reader: this cannot be trusted to describe a run. A
    `--status` that raised on a half-written file would be worse than one that
    admits it cannot tell.
    """
    try:
        raw = json.loads(path.read_text())
        return Heartbeat(
            run_id=str(raw["run_id"]),
            stage=str(raw["stage"]),
            done=int(raw["done"]),
            total=int(raw["total"]),
            started_at=datetime.fromisoformat(str(raw["started_at"])),
            updated_at=datetime.fromisoformat(str(raw["updated_at"])),
            in_flight=_read_item(raw.get("in_flight")),
            last_completed=_read_item(raw.get("last_completed")),
            deadline=(
                datetime.fromisoformat(str(raw["deadline"]))
                if raw.get("deadline") is not None
                else None
            ),
        )
    except (OSError, ValueError, KeyError, TypeError):
        return None
