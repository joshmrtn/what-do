"""Per-item progress, and the policy that decides how much of it is said.

The stage reports every item and decides nothing else. What that becomes — a
log line every so often, a heartbeat file a reader can interrogate — is a
policy the orchestrator wires, in one place, because it is the part that has to
change when extraction stops taking minutes an event and starts taking seconds.

Not to be confused with `presentation/progress.py`, which is the CLI spinner.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Protocol

#: An item is about to be worked. Carries the item, not a count: nothing has
#: been finished, and this is the phase that names what a killed batch died on.
STARTED = "started"
#: An item is done, and `done` includes it.
FINISHED = "finished"


@dataclass(frozen=True)
class Progress:
    """One item's worth of "k of n, at t", from a stage that is mid-pass.

    Attributes:
        stage: Which stage is reporting, as it should read in a log line.
        done: Items finished. Includes this one on `FINISHED`, excludes it on
            `STARTED` — an item that has begun has finished nothing.
        total: The size of the queue, known before the first item is worked.
        item_id: Stable identifier, so a death can be traced to a row.
        label: What a person would call it.
        phase: `STARTED` or `FINISHED`.
        now: From the stage's injected clock. Never read from a sink — a sink
            that asks the wall clock is a sink that cannot be tested.
        deadline: When the stage must stop starting work, or None when nothing
            bounds it. Only extraction has one today.
    """

    stage: str
    done: int
    total: int
    item_id: str
    label: str
    phase: str
    now: datetime
    deadline: datetime | None = None


class ProgressFn(Protocol):
    """What a stage is handed. Deliberately one argument and no return."""

    def __call__(self, report: Progress) -> None:
        ...


def format_duration(span: timedelta) -> str:
    """A span in the unit that reads honestly at that scale.

    Seconds below a minute, minutes below an hour, `2h05m` above — an eight
    hour batch reported as `481m` is arithmetic the reader has to do.
    """
    seconds = max(0, int(span.total_seconds()))
    if seconds < 60:
        return f"{seconds}s"
    minutes = seconds // 60
    if minutes < 60:
        return f"{minutes}m"
    return f"{minutes // 60}h{minutes % 60:02d}m"


class ProgressLog:
    """Emits a progress line on a milestone, or after too long a silence.

    Two rules because one does not survive the provider swap. Milestones alone
    put two hours between lines on a local-Ollama extraction, which reads as a
    dead batch; a time floor alone writes a hundred lines against a hosted API
    that finishes the same queue in an hour. Together, whichever fires first,
    the volume stays around a dozen lines at either speed.

    **The heartbeat is driven by items finishing, not by a timer.** A stage
    whose model call never returns reports nothing at all, so it goes quiet
    here. That is what `--status` and its stall state are for; a thread ticking
    in the background to report on a hung one would be a worse trade.
    """

    def __init__(
        self,
        logger: Any,
        *,
        milestone_fraction: float,
        heartbeat: timedelta,
    ) -> None:
        self._logger = logger
        self._fraction = milestone_fraction
        self._heartbeat = heartbeat
        self._started_at: datetime | None = None
        self._last_line: datetime | None = None
        self._next_milestone = 0
        self._last_done = 0

    def __call__(self, report: Progress) -> None:
        """Take one report, and say something about it or not."""
        if self._is_new_pass(report):
            self._begin(report)
        self._last_done = report.done

        if report.phase != FINISHED:
            return

        milestone = report.done >= self._next_milestone
        silent = (
            self._last_line is not None
            and report.now - self._last_line >= self._heartbeat
        )
        # `or`, then one line. A milestone landing inside a silence is one
        # event, and reporting it twice is exactly the noise this rations.
        if not (milestone or silent):
            return

        while report.done >= self._next_milestone:
            self._next_milestone += self._step(report.total)
        self._last_line = report.now
        self._emit(report)

    def _is_new_pass(self, report: Progress) -> bool:
        """Whether this report belongs to a pass that has not been seen.

        A pass opens with its first item starting, having finished none — and
        `done` only ever climbs within one, so a count that drops is a fresh
        pass too. It matters because a rescore drives the embedding stage again
        in the same process, hours later, and carrying the first pass's start
        forward would report an elapsed time that includes the gap.
        """
        if self._started_at is None or report.done < self._last_done:
            return True
        return report.phase == STARTED and report.done == 0

    def _begin(self, report: Progress) -> None:
        self._started_at = report.now
        # Set rather than left None, so the first heartbeat is measured from
        # the stage starting rather than from the first line it happens to
        # write — otherwise nothing is due until the first milestone, which on
        # a long queue is hours away.
        self._last_line = report.now
        self._next_milestone = self._step(report.total)
        self._last_done = 0

    def _step(self, total: int) -> int:
        """How many items lie between milestones. At least one."""
        return max(1, round(total * self._fraction))

    def _emit(self, report: Progress) -> None:
        started = self._started_at if self._started_at is not None else report.now
        elapsed = report.now - started
        percent = int(round(100 * report.done / report.total)) if report.total else 100
        parts = [
            f"{report.stage} {report.done}/{report.total} ({percent}%)",
            f"{format_duration(elapsed)} elapsed",
        ]
        if report.done:
            each = elapsed.total_seconds() / report.done
            parts.append(f"{each:.0f}s each" if each >= 1 else f"{each:.1f}s each")
        if report.deadline is not None:
            remaining = report.deadline - report.now
            parts.append(
                f"{format_duration(remaining)} of budget left"
                if remaining > timedelta(0)
                else "budget spent"
            )
        self._logger.info(
            " · ".join(parts), component=report.stage, duration_ms=0
        )
