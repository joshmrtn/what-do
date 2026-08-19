"""`what-do --status`: is a batch running, how far along, and is anything wrong.

Four states, from two signals read together. The lock is the only one that dies
with the process; the heartbeat is the only one that knows what the process was
doing. Either alone can distinguish two states, and the pair distinguishes all
four — of which two were invisible before this existed:

| lock | heartbeat | state |
|---|---|---|
| held | moving | **running** |
| held | unchanged past the stall threshold | **stalled** |
| free | present, its run still open | **died**, naming the event |
| free | absent, or its run completed | **idle** |
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone, tzinfo
from pathlib import Path

from src.models.run import RunRecord
from src.observability.heartbeat import (
    HEARTBEAT_PATH,
    LOCK_PATH,
    Heartbeat,
    read_heartbeat,
)
from src.observability.lock import batch_lock_held
from src.observability.reporter import format_duration
from src.storage.protocols import RunRepository

#: The width of the state word, so the detail beneath it lines up under one
#: another rather than under four different indents.
_INDENT = " " * 12


@dataclass(frozen=True)
class StatusInputs:
    """Everything `render_status` needs, gathered in one read.

    A value rather than the repositories themselves, so the rendering can be
    exercised over states that are awkward to arrange for real — a batch that
    died half an hour ago, a heartbeat whose run has since completed.
    """

    lock_held: bool
    heartbeat: Heartbeat | None
    heartbeat_run: RunRecord | None
    open_run: RunRecord | None
    latest_run: RunRecord | None


@dataclass(frozen=True)
class StatusReport:
    """What to print, and what printing it has just told someone.

    The second half exists because the crash footnote clears once it has been
    read, which needs a write — and `render_status` is a pure function that
    should stay one. It names the decision instead of taking the action, so the
    caller can record it without re-deriving *did the note appear*, a condition
    that would then live in two places and drift.
    """

    text: str
    #: The run whose death this text has just footnoted, if it did. The caller
    #: marks it reported, **after** printing: a crash between the two must lose
    #: the stamp rather than the notice.
    crash_reported: str | None = None


def probe_status(
    runs: RunRepository | None,
    *,
    lock_path: Path = LOCK_PATH,
    heartbeat_path: Path = HEARTBEAT_PATH,
) -> StatusInputs:
    """Read the lock, the heartbeat and the run rows, in that order.

    The lock first, deliberately: it is the fastest-moving of the three, so
    reading it first makes the window in which the picture could be internally
    inconsistent as small as it can be.

    Args:
        runs: The run history, or None when there is no database to read one
            from — `--status` is exactly what someone runs before the first
            batch has ever finished, and a lock and a heartbeat still answer
            most of the question without it.
        lock_path: The batch lock.
        heartbeat_path: The live state file.
    """
    lock_held = batch_lock_held(lock_path)
    heartbeat = read_heartbeat(heartbeat_path)
    if runs is None:
        return StatusInputs(
            lock_held=lock_held, heartbeat=heartbeat,
            heartbeat_run=None, open_run=None, latest_run=None,
        )
    return StatusInputs(
        lock_held=lock_held,
        heartbeat=heartbeat,
        heartbeat_run=(
            runs.get(heartbeat.run_id) if heartbeat is not None else None
        ),
        open_run=runs.open_run(),
        latest_run=runs.latest(),
    )


def render_status(
    *,
    lock_held: bool,
    heartbeat: Heartbeat | None,
    heartbeat_run: RunRecord | None,
    open_run: RunRecord | None,
    latest_run: RunRecord | None,
    now: datetime,
    stall_after: timedelta,
    zone: tzinfo = timezone.utc,
) -> StatusReport:
    """Describe what the batch is doing, in the terms a person would ask it.

    Args:
        lock_held: Whether any process holds the batch lock.
        heartbeat: The live state file, or None when there is none to read —
            which covers an absent file and an unreadable one alike, because
            both mean the same thing to a reader.
        heartbeat_run: The `run_history` row the heartbeat names, if any. What
            tells a leftover from a death: a heartbeat whose run has a
            `completed_at` is the remains of a batch that finished and could
            not clean up.
        open_run: The newest run that began and never finished. Evidence of a
            death that happened before extraction ever reported anything.
        latest_run: The newest run row at all, for the idle case.
        now: Injected clock.
        stall_after: How long an item may be in flight before it is stuck.
            zone: What zone to print clock times in. Timestamps are stored in
            UTC and the batch runs at 02:00 local, so a line reading `06:00`
            for it is the kind of wrong that looks right.

    Returns:
        The multi-line summary, and any crash the summary has just footnoted.
        Never raises on partial information — every unknown reads as an unknown
        rather than as an alarm.
    """
    live = heartbeat if _is_live(heartbeat, heartbeat_run) else None

    if lock_held:
        if live is None:
            return StatusReport(
                _lines("running", ["no progress reported yet — still starting up"])
            )
        stalled_for = _stalled_for(live, now, stall_after)
        if stalled_for is not None:
            return StatusReport(_lines("stalled", _stall_detail(live, stalled_for)))
        return StatusReport(_lines("running", _progress_detail(live, now)))

    if live is not None:
        return StatusReport(_lines("died", _death_detail(live, now, zone)))
    if _died_without_reporting(open_run, latest_run):
        return StatusReport(_lines("died", [
            "a run is open in run_history and no process holds the lock,",
            "and it died before it reported any progress — see logs/batch-latest.log",
        ]))
    detail, footnoted = _idle_detail(latest_run, open_run, now, zone)
    return StatusReport(_lines("idle", detail), crash_reported=footnoted)


def _is_live(heartbeat: Heartbeat | None, run: RunRecord | None) -> bool:
    """Whether this heartbeat describes a run that is still open.

    Belt and braces behind the batch removing the file on a clean exit: only
    the cross-check survives a crash *during* cleanup, and reading a leftover
    as a death would cry wolf every morning.
    """
    return heartbeat is not None and run is not None and run.completed_at is None


def running_note(
    heartbeat: Heartbeat | None,
    *,
    now: datetime,
    fresh_within: timedelta,
) -> str | None:
    """A phrase for the stale banner, or None when no batch is working.

    Reads the heartbeat and deliberately not the lock. `--status` is an
    explicit question and can afford the probe; the listing is the hot path,
    run many times a day, and every probe holds the batch lock for a few
    microseconds. Freshness answers this question well enough — a heartbeat
    updated a minute ago is a batch that is working, whatever the lock says,
    and one that has not moved in an hour is a file left behind.

    Args:
        heartbeat: The live state file, or None.
        now: Injected clock.
        fresh_within: How recently it must have moved to count as alive. The
            stall threshold, because that is already the answer to "how long
            may this go quiet and still be working".
    """
    if heartbeat is None or now - heartbeat.updated_at >= fresh_within:
        return None
    note = _headline(heartbeat)
    if heartbeat.deadline is not None and heartbeat.deadline > now:
        note += f", ~{format_duration(heartbeat.deadline - now)} of budget left"
    return note


def _died_without_reporting(
    open_run: RunRecord | None, latest_run: RunRecord | None
) -> bool:
    """Whether the open run is the run that just happened.

    `open_run` returns the newest *unfinished* row whatever ran after it, which
    is right for its own question — *has a crash gone unexamined* — and wrong
    for this one. The live database's open run is 2026-08-12, six successful
    runs ago; reporting that as a death every morning is an alarm that can
    never clear, and one nobody reads. It is still named in the idle line.
    """
    if open_run is None:
        return False
    return latest_run is None or latest_run.run_id == open_run.run_id


def _lines(state: str, detail: list[str]) -> str:
    head, *rest = detail
    out = [f"{state:<9} ·  {head}"]
    out.extend(f"{_INDENT}{line}" for line in rest)
    return "\n".join(out)


def _headline(beat: Heartbeat) -> str:
    percent = int(round(100 * beat.done / beat.total)) if beat.total else 100
    return f"{beat.stage} {beat.done}/{beat.total} ({percent}%)"


def _rate(beat: Heartbeat, now: datetime) -> float | None:
    """Seconds an item so far, or None while nothing has finished."""
    if beat.done <= 0:
        return None
    return (now - beat.started_at).total_seconds() / beat.done


def _progress_detail(beat: Heartbeat, now: datetime) -> list[str]:
    elapsed = now - beat.started_at
    rate = _rate(beat, now)
    pace = [f"{format_duration(elapsed)} elapsed"]
    if rate is not None:
        pace.append(f"{rate:.0f}s each")
    if beat.deadline is not None:
        remaining = beat.deadline - now
        pace.append(
            f"{format_duration(remaining)} of budget left"
            if remaining > timedelta(0)
            else "budget spent"
        )
    detail = [_headline(beat), " · ".join(pace)]
    projection = _projection(beat, now, rate)
    if projection is not None:
        detail.append(projection)
    if beat.in_flight is not None:
        detail.append(
            f"now: {beat.in_flight.label} "
            f"({format_duration(now - beat.in_flight.since)} in flight)"
        )
    return detail


def _projection(beat: Heartbeat, now: datetime, rate: float | None) -> str | None:
    """Which of the two bounds runs out first, and what that leaves undone.

    The same arithmetic answers both providers. Against local Ollama the budget
    binds and the queue is the thing that does not get finished; against a
    hosted API the queue finishes with the budget barely touched, and saying
    "on course" is the whole answer.
    """
    if rate is None or rate <= 0:
        return None
    left = beat.total - beat.done
    if left <= 0:
        return None
    needed = timedelta(seconds=rate * left)
    if beat.deadline is None:
        return f"~{format_duration(needed)} to finish the queue at this rate"
    affordable = int(max(0.0, (beat.deadline - now).total_seconds()) // rate)
    if affordable >= left:
        return (
            f"on course to finish the queue, with "
            f"{format_duration(beat.deadline - now - needed)} of budget to spare"
        )
    return (
        f"budget binds first: ~{affordable} more this run, "
        f"~{left - affordable} deferred to tomorrow"
    )


def _stalled_for(
    beat: Heartbeat, now: datetime, stall_after: timedelta
) -> timedelta | None:
    """How long the in-flight item has been in flight, once that is too long.

    Nothing in flight is never stalled: between items the stage is choosing its
    next one, which takes no time at all.
    """
    if beat.in_flight is None:
        return None
    waited = now - beat.in_flight.since
    return waited if waited >= stall_after else None


def _stall_detail(beat: Heartbeat, waited: timedelta) -> list[str]:
    assert beat.in_flight is not None  # `_stalled_for` returned a duration
    return [
        _headline(beat),
        f"{beat.in_flight.label} has been in flight for {format_duration(waited)}",
        f"the lock is still held, so the process is alive — {beat.in_flight.item_id}",
    ]


def _death_detail(beat: Heartbeat, now: datetime, zone: tzinfo) -> list[str]:
    """What it was doing when it stopped, which is the point of the file.

    The in-flight item is the one the transcript cannot name: every call is
    recorded once it returns or raises, so an event the process was killed
    inside appears in no other artefact.
    """
    detail = [_headline(beat), "no process holds the lock, and its run is still open"]
    if beat.in_flight is not None:
        detail.append(
            f"died on: {beat.in_flight.label} ({beat.in_flight.item_id}), "
            f"in flight since {beat.in_flight.since.astimezone(zone):%H:%M}"
        )
    elif beat.last_completed is not None:
        detail.append(
            f"last finished: {beat.last_completed.label} "
            f"({beat.last_completed.item_id}) at "
            f"{beat.last_completed.since.astimezone(zone):%H:%M}"
        )
    detail.append(f"last seen {format_duration(now - beat.updated_at)} ago")
    return detail


def _idle_detail(
    latest: RunRecord | None,
    open_run: RunRecord | None,
    now: datetime,
    zone: tzinfo,
) -> tuple[list[str], str | None]:
    """The idle lines, and the crash they footnote if they footnote one."""
    if latest is None:
        return ["no batch has ever run against this database"], None
    if latest.completed_at is None:
        return [f"a run started {format_duration(now - latest.started_at)} ago"], None
    detail = [
        f"last run {latest.started_at.astimezone(zone):%Y-%m-%d %H:%M} → "
        f"{latest.completed_at.astimezone(zone):%H:%M}, "
        f"{latest.outcome}, {format_duration(now - latest.completed_at)} ago"
    ]
    detail.extend(f"  {error}" for error in latest.errors)
    crash = _worth_footnoting(open_run, latest)
    if crash is None:
        return detail, None
    # Named, not hidden: `open_run` exists so an unexamined crash cannot be
    # lost. A footnote is where it belongs once later runs have succeeded — and
    # printing it is what makes it read, so it is reported back to be stamped.
    detail.append(
        f"note: the run of {crash.started_at.astimezone(zone):%Y-%m-%d} "
        "never finished"
    )
    return detail, crash.run_id


def _worth_footnoting(
    open_run: RunRecord | None, latest: RunRecord
) -> RunRecord | None:
    """The unfinished run still worth mentioning, if there is one.

    Two conditions, and they answer different questions. The run must not be
    the latest one — while it is, the states above report it directly and
    unconditionally, so a footnote would be the same news twice. And it must
    not already have been reported: a death examined once has been examined,
    and a week-old crash still footnoting a run of successes is noise rather
    than a warning.

    Any *completed* later run clears it, whatever its outcome. When the latest
    run is `partial` or `failed`, the line above already carries that outcome
    and its errors — adding an older death underneath reports a stale problem
    on a display that is describing a current one. Requiring `success` would
    also pin the note for as long as a bad patch lasted, which is the
    stuck-forever behaviour this exists to remove.
    """
    if open_run is None or open_run.run_id == latest.run_id:
        return None
    return open_run if open_run.crash_reported_at is None else None
