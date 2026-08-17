"""Whether a read-time rescore should run, and why not when it should not.

Kept apart from the recomputation itself. Most invocations decide "no" and must
cost nothing to do so, and the reasons are worth naming rather than collapsing
into a boolean — two of them are about *not making things worse*, and the caller
says different things about each.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta

#: Recompute: the run is tonight's, no batch is writing, and the forecast has
#: aged past the read TTL.
RESCORE = "rescore"
#: A batch is in flight. Not merely "later" — writing beside it is forbidden.
BATCH_RUNNING = "batch_running"
#: The stored ranking is for a different night. Rescoring it would apply
#: tonight's forecast under that night's scope.
NOT_TONIGHT = "not_tonight"
#: Nothing to gain. Either the forecast is inside the TTL, or there is none.
FORECAST_FRESH = "forecast_fresh"


def should_rescore(
    *,
    run_date: date,
    tonight: date,
    forecast_issued_at: datetime | None,
    now: datetime,
    ttl: timedelta,
    batch_running: bool,
) -> str:
    """Decide whether to recompute the stored ranking before showing it.

    The order of the checks is the design. `batch_running` comes first because
    it is the only one where proceeding is actively harmful: `connect()` sets
    `busy_timeout`, so a write against a database an eight-hour batch is holding
    does not fail — it *waits*, turning an instant listing into a hung one.

    `run_date` comes next because it is a correctness bound, not an
    optimisation. `scope_filter` derives its floor and horizon from the run
    date, so rescoring an older run would judge tonight's forecast against that
    night's scope.

    Only then does freshness decide, which is what makes a second invocation
    seconds after the first cost nothing.

    Args:
        run_date: The run the listing was loaded from.
        tonight: The night being shown, in the view's own zone.
        forecast_issued_at: When the ranking's freshest forecast was issued, or
            None when no event carries one.
        now: The current instant.
        ttl: How old a forecast may be before it is worth refreshing.
        batch_running: Whether a batch is currently in flight.

    Returns:
        `RESCORE`, or the constant naming why not.
    """
    if batch_running:
        return BATCH_RUNNING
    if run_date != tonight:
        return NOT_TONIGHT
    if forecast_issued_at is None:
        return FORECAST_FRESH
    # A stamp in the future is clock skew. Reading it as a large age would
    # trigger a write on nothing at all.
    if now - forecast_issued_at <= ttl:
        return FORECAST_FRESH
    return RESCORE


#: Beyond this, an unfinished run is a crash rather than a batch in flight.
#: A run takes roughly eight hours and only `finish` closes the row, so a killed
#: process leaves `completed_at` NULL forever — and reading that as "a batch is
#: running" would disable the rescore permanently, silently, with the fix being
#: a database edit nobody would know to make.
BATCH_ASSUMED_DEAD_AFTER = timedelta(hours=18)


def batch_is_running(open_run_started_at: datetime | None, now: datetime) -> bool:
    """Whether a batch is genuinely in flight right now.

    Args:
        open_run_started_at: When the open run began, or None if there is none.
        now: The current instant.

    Returns:
        True only for a run recent enough to still be alive.
    """
    if open_run_started_at is None:
        return False
    return now - open_run_started_at < BATCH_ASSUMED_DEAD_AFTER
