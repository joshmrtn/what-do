"""CLI entry points for the what-do application.

Reads precomputed rows and renders them. No LLM, no scoring and no reordering
happen here — the batch already decided, and this is the view onto that
decision. Query time may reach the network for something cheap and perishable;
it may never reach a model, because that is the difference between milliseconds
and minutes.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta, tzinfo
from pathlib import Path
from typing import Any, Callable, TextIO
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import yaml


from src.config import (
    ViewConfig,
    DEFAULT_DAY_STARTS_AT,
    DEFAULT_DISLIKES_PATH,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_LIKES_PATH,
    ConfigError,
    load_config,
)
from src.models.event import Event
from src.presentation.freshness import (
    freshness_notice,
    latest_forecast,
    preference_state,
)
from src.presentation.handles import HANDLE_SIGIL, short_handle
from src.scoring.preference_revision import hash_preference_files
from src.presentation.filters import (
    RankedEvent,
    after_sunset,
    during_night,
    matching,
    night_of,
    overlapping,
    parse_time_window,
)
from src.presentation.render import (
    render_explanation,
    render_raw,
    render_recommendations,
    staleness_notice,
)
from src.storage.sqlite.connection import DEFAULT_DB_PATH, has_schema
from src.composition.storage import build_view_storage
from src.storage.queries import load_ranked_events

_DEFAULT_SEEDS_PATH = Path("data/seeds.yaml")

_NO_DATABASE_MESSAGE = "No database yet — run the overnight batch to build one."
_NO_RECOMMENDATIONS_MESSAGE = "No recommendations yet — run the overnight batch to populate them."

PairLoader = Callable[..., list[RankedEvent]]


def _default_pairs(
    db_path: Path,
    run_date: date | None = None,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
) -> list[RankedEvent]:
    """Join the three stores this view reads into one listing.

    Storage comes from the shared factory rather than being wired here: this
    used to construct its own repositories and took the default embedding
    model while the batch passed the configured one, so the two agreed only
    while `config.yaml` happened to name the default.
    """
    storage = build_view_storage(db_path, embedding_model)
    return load_ranked_events(
        storage.events, storage.scores, storage.rankings, run_date=run_date
    )
def _default_events(
    db_path: Path,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
) -> list[Event]:
    """Every stored event, superseded ones included, for `--raw`.

    Through the repository like every other read. `--raw` used to call
    `storage.events.load_events` directly — the one path that does not filter
    supersession — and it was a *default argument*, so production took the
    unfiltered path while every test injected its own. That is the shape the
    footgun table names: a default only production reaches is untested by
    construction.

    `include_superseded=True` is deliberate rather than incidental. `--raw`
    means every stored event, and a merge worth disagreeing with is exactly
    what this view is for; the renderer marks them so nobody reads a
    merged-away duplicate as real.
    """
    return build_view_storage(db_path, embedding_model).events.load_all(
        include_superseded=True
    )


EventLoader = Callable[..., list[Event]]
ReadinessCheck = Callable[[Path], bool]
#: Reads how stale the listing's inputs are. One callable rather than three,
#: because a test substituting it should not have to own a database, a clock and
#: two files on disk to say "nothing is stale".
FreshnessProbe = Callable[..., str | None]


def _default_freshness(
    db_path: Path,
    embedding_model: str,
    *,
    events: list[Event],
    now: datetime,
    ttl: timedelta,
    likes_path: Path,
    dislikes_path: Path,
) -> str | None:
    """Whether the forecast has aged or the preference files have moved.

    Reads through the repository like every other view read, and embeds
    nothing: the whole point is that asking is cheap enough to do on every
    invocation.

    The preference paths are parameters rather than the module constants read
    directly, so this function is reachable from a test without owning two files
    at fixed locations. A default only production takes is untested by
    construction, and this one would have read the developer's own `likes.txt`
    on every run of the CLI suite.
    """
    revisions = build_view_storage(db_path, embedding_model).preference_revisions
    return freshness_notice(
        forecast_issued_at=latest_forecast(events),
        now=now,
        ttl=ttl,
        preferences=preference_state(
            hash_preference_files(likes_path, dislikes_path),
            revisions.latest(),
        ),
    )


@dataclass(frozen=True)
class ViewSettings:
    """What the CLI needs from config to say which night it is showing.

    Attributes:
        zone: The timezone events are judged in. From `config.location`, which
            derives it from the coordinates rather than from the machine.
        day_starts_at: Local time of day at which the listing rolls over.
        source_urls: Human-facing page per `source_type`, for events that carry
            no URL of their own. Empty when config is unreadable — attribution is
            a convenience, and losing it must never cost the listing.
        embedding_model: Which model's vectors the batch wrote, so the view
            reads under the same name. Falls back to the default when config is
            unreadable — the same degraded path as the zone, and announced by
            the same warning.
        view: How many events a listing shows, how far `--upcoming` reaches,
            how many reasons appear, and when a span becomes a daily programme.
            Falls back to the dataclass defaults when config is unreadable, on
            the same terms as the zone — a listing with sensible numbers beats
            no listing.
        warning: Emitted to stderr when the settings had to be guessed. Carried
            on the value rather than printed by the loader, so the loader stays
            substitutable in tests without capturing a stream.
    """

    zone: tzinfo
    day_starts_at: time
    source_urls: Mapping[str, str] = field(default_factory=dict)
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    view: ViewConfig = field(default_factory=ViewConfig)
    warning: str | None = None


ViewSettingsLoader = Callable[[], ViewSettings]


def default_view_settings() -> ViewSettings:
    """Read the view's zone and rollover hour from config, degrading if absent.

    A missing or invalid `config.yaml` is a normal state for a fresh clone, and
    a CLI that dies over one is worse than a CLI showing the system's day. The
    guess is announced rather than made quietly, because a listing headed with
    the wrong date is exactly the kind of wrong that looks right.
    """
    try:
        config = load_config()
        return ViewSettings(
            zone=ZoneInfo(config.location.timezone),
            day_starts_at=config.day_starts_at,
            source_urls=config.sources.site_url_by_source_type(),
            embedding_model=config.models.embeddings,
            view=config.view,
        )
    except (ConfigError, OSError, ZoneInfoNotFoundError) as exc:
        system_zone = datetime.now().astimezone().tzinfo
        return ViewSettings(
            zone=system_zone if system_zone is not None else ZoneInfo("UTC"),
            day_starts_at=DEFAULT_DAY_STARTS_AT,
            warning=f"Warning: no usable config ({exc}) — using the system timezone.",
        )


def _load_seeds_raw(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"handles": [], "venues": []}
    with open(path) as f:
        return yaml.safe_load(f) or {"handles": [], "venues": []}


def _write_seeds(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True)


def _cmd_add_source(args: argparse.Namespace, stdout: TextIO, stderr: TextIO) -> int:
    seeds_path = Path(args.seeds_file) if args.seeds_file else _DEFAULT_SEEDS_PATH
    data = _load_seeds_raw(seeds_path)

    # Stripped before the truthiness test: a whitespace-only handle is falsy
    # to a person and truthy to Python, so `add-source '   '` wrote '@   ' into
    # seeds.yaml and reported success — a real entry that discovery would fetch.
    handle_arg = args.handle.strip() if args.handle else ""
    venue_arg = args.venue.strip() if args.venue else ""
    address_arg = args.address.strip() if args.address else ""

    if handle_arg:
        handle = handle_arg if handle_arg.startswith("@") else f"@{handle_arg}"
        if handle in data.get("handles", []):
            print(f"{handle} is already in seeds.yaml", file=stdout)
            return 0
        data.setdefault("handles", []).append(handle)
        _write_seeds(seeds_path, data)
        print(f"Added {handle} to seeds.yaml", file=stdout)
        return 0

    if venue_arg and address_arg:
        venues = data.get("venues", [])
        for v in venues:
            if v.get("name") == venue_arg:
                print(f"Venue '{venue_arg}' is already in seeds.yaml", file=stdout)
                return 0
        venues.append({"name": venue_arg, "address": address_arg})
        data["venues"] = venues
        _write_seeds(seeds_path, data)
        print(f"Added venue '{venue_arg}' to seeds.yaml", file=stdout)
        return 0

    print("Error: provide --handle or both --venue and --address", file=stderr)
    return 1


def _cmd_recommend(
    args: argparse.Namespace,
    *,
    get_now: Callable[[], datetime],
    stdout: TextIO,
    stderr: TextIO,
    load_pairs: PairLoader,
    load_all_events: EventLoader,
    db_ready: ReadinessCheck,
    view: ViewSettings,
    check_freshness: FreshnessProbe,
) -> int:
    """Render the default view: the latest run, filtered to tonight."""
    if (conflict := _conflicting_flags(args)) is not None:
        print(f"Error: {conflict}", file=stderr)
        return 1

    if args.all and args.limit is not None:
        # Served rather than refused: the request is coherent, one half simply
        # cannot apply. Announced, because a silently dropped flag is how
        # `--time` came to render an unfiltered listing that looked filtered.
        print("Warning: --all shows every event, so --limit is ignored.", file=stderr)

    db_path = Path(args.db) if args.db else DEFAULT_DB_PATH

    if not db_ready(db_path):
        # An uninitialised default path is the state before the first batch has
        # ever run, which is normal. A missing *named* one is a typo, and
        # reporting that as "no events" would hide the mistake.
        if args.db:
            print(f"Error: no database at {db_path}", file=stderr)
            return 1
        print(_NO_DATABASE_MESSAGE, file=stdout)
        return 0

    if args.raw:
        print(
            render_raw(load_all_events(db_path, embedding_model=view.embedding_model)),
            file=stdout,
            end="",
        )
        return 0

    if args.limit is not None and args.limit < 1:
        # Not `or default`: 0 is falsy, so asking for no events silently
        # produced the default, and a negative slices from the end —
        # `pairs[:-5]` drops the last five and renders a plausible, wrong list.
        print(f"Error: --limit must be 1 or more, got {args.limit}", file=stderr)
        return 1

    # A blank value is not a value. Each of these is read with `if args.x:`
    # somewhere, and an empty string is falsy — so a flag the user typed reads
    # as one they did not, and the command answers a different question.
    for flag, value, wants in (
        ("--explain", args.explain, "an event: a #handle, or part of a title"),
        ("--time", args.time, "a window, e.g. 20:30-23:30"),
        ("--date", args.date, "a date as YYYY-MM-DD"),
        ("--run-date", args.run_date, "a date as YYYY-MM-DD"),
        ("--db", args.db, "a path"),
    ):
        if value is not None and not value.strip():
            print(f"Error: {flag} needs {wants}", file=stderr)
            return 1

    try:
        run_date = date.fromisoformat(args.run_date) if args.run_date else None
    except ValueError:
        print(f"Error: --run-date must be YYYY-MM-DD, got {args.run_date!r}", file=stderr)
        return 1

    if args.explain is not None:
        return _cmd_explain(
            args.explain,
            db_path=db_path,
            run_date=run_date,
            load_pairs=load_pairs,
            load_all_events=load_all_events,
            stdout=stdout,
            stderr=stderr,
            view=view,
        )

    try:
        window = parse_time_window(args.time) if args.time else None
    except ValueError as exc:
        print(f"Error: {exc}", file=stderr)
        return 1

    pairs = load_pairs(
        db_path, run_date=run_date, embedding_model=view.embedding_model
    )
    if not pairs:
        print(_NO_RECOMMENDATIONS_MESSAGE, file=stdout)
        return 0

    # Warn only here, where the guessed zone actually decides something. The
    # `--raw` and empty-database paths never consult it, and announcing a
    # fallback that changed no output is noise on the one screen a user sees
    # before they have set anything up.
    if view.warning:
        print(view.warning, file=stderr)

    # The night is named in the view's own zone, not the machine's: event start
    # times are aware in it, and taking the date from wherever the VM thinks it
    # is would compare them against a different day.
    tonight = night_of(get_now().astimezone(view.zone), view.day_starts_at)
    # Before anything is rendered, and to stderr like every other warning, so a
    # piped listing is unchanged and a terminal one cannot miss it. The batch
    # that fails is exactly the batch that leaves an older ranking in place.
    notice = staleness_notice(pairs[0].ranking.run_date, tonight)
    if notice is not None:
        print(notice, file=stderr)

    # A different question from the one above, and worth asking separately: that
    # one is about a batch that never ran, this is about a batch that ran and
    # whose inputs have since moved. Both go to stderr for the same reason.
    freshness = check_freshness(
        db_path,
        view.embedding_model,
        events=[pair.event for pair in pairs],
        now=get_now(),
        ttl=timedelta(minutes=view.view.refresh_ttl_minutes),
        likes_path=DEFAULT_LIKES_PATH,
        dislikes_path=DEFAULT_DISLIKES_PATH,
    )
    if freshness is not None:
        print(freshness, file=stderr)

    if args.upcoming is not None:
        return _cmd_upcoming(
            args,
            pairs=pairs,
            tonight=tonight,
            window=window,
            stdout=stdout,
            stderr=stderr,
            view=view,
        )

    try:
        nights = _nights_to_show(args, tonight)
    except ValueError as exc:
        print(f"Error: {exc}", file=stderr)
        return 1
    sections = []
    for night in nights:
        # `during_night` drops undated events, so they are re-added rather than
        # filtered: a missing start time is a gap in what we know, not evidence
        # about when. Only on tonight's section — an undated event belongs to
        # the night you are standing in, not to every night listed.
        selected = during_night(pairs, night, view.day_starts_at, view.zone)
        if night == tonight:
            selected = selected + [p for p in pairs if p.event.start_time is None]
        selected.sort(key=lambda ranked: ranked.rank)

        if window is not None:
            selected = overlapping(
                selected,
                *window,
                night=night,
                long_span_hours=view.view.long_span_hours,
            )
        if args.after_sunset:
            selected = after_sunset(selected)

        sections.append(
            render_recommendations(
                selected,
                heading=night.strftime("%A %-d %B"),
                verbose=args.verbose,
                limit=None if args.all else (args.limit if args.limit is not None else view.view.limit),
                color=_supports_color(stdout),
                source_urls=view.source_urls,
                reason_limit=view.view.reason_limit,
            )
        )

    print(
        "\n".join(sections),
        file=stdout,
        end="",
    )
    return 0


def _supports_color(stream: TextIO) -> bool:
    """ANSI only when the destination is a terminal, so pipes stay clean."""
    return bool(getattr(stream, "isatty", lambda: False)())


#: What a bare `--upcoming` records, so the configured default can be applied
#: once `view` is in hand — the parser is built before config is read.
#:
#: An **object**, not a number. argparse applies `type=` only to strings, so a
#: non-string `const` passes through untouched, which is what keeps it outside
#: the value domain entirely. Every integer sentinel is a value someone can
#: type: `0` made `--upcoming 0` silently mean "use the default", and `-1` did
#: exactly the same thing once it replaced it — argparse accepts a negative
#: value when no option looks like a negative number.
_UPCOMING_FROM_CONFIG = object()


def _cmd_upcoming(
    args: argparse.Namespace,
    *,
    pairs: list[RankedEvent],
    tonight: date,
    window: tuple[time, time] | None,
    stdout: TextIO,
    stderr: TextIO,
    view: ViewSettings,
) -> int:
    """One ranked list across the nights *after* tonight, best first.

    Not `--days`, which is per-night sections. The question here is "what is
    coming up that I should plan for" — anything needing a booking, a table, a
    drive, or a ticket bought before it sells out. A great event three weeks out
    is ranked and stored and, until now, invisible until the morning of.

    This is a filter rather than a re-ranking, and only because scores are never
    normalised per batch: a score from next Friday sorts against one from
    tonight honestly. It is also what the 90-day horizon was raised for.
    """
    if args.upcoming is _UPCOMING_FROM_CONFIG:
        args.upcoming = view.view.upcoming_days
    if args.upcoming < 1:
        print(f"Error: --upcoming must be 1 or more, got {args.upcoming}", file=stderr)
        return 1

    # Starts tomorrow. `--upcoming` reads as "after tonight", the default view
    # already answers tonight, and tonight is the one night needing no planning
    # — it is also the busiest, so including it crowds out the far-out events
    # this exists to surface.
    first = tonight + timedelta(days=1)
    last = first + timedelta(days=args.upcoming - 1)
    selected = [
        pair
        for pair in pairs
        if pair.event.start_time is not None
        and first <= pair.event.start_time.astimezone(view.zone).date() <= last
    ]
    if window is not None:
        # Grouped by the night each event falls on, because a time-of-day
        # window has to be anchored somewhere and this view spans many nights.
        # One shared anchor would compare a Saturday 20:00 event against
        # tonight's window — the bug this replaced, in a new place.
        #
        # `overlapping` keeps its single-night contract; the view that knows it
        # has many does the grouping, rather than the filter learning to guess.
        by_night: dict[date, list[RankedEvent]] = {}
        for pair in selected:
            assert pair.event.start_time is not None  # filtered above
            night = night_of(
                pair.event.start_time.astimezone(view.zone), view.day_starts_at
            )
            by_night.setdefault(night, []).append(pair)
        selected = [
            pair
            for night, group in by_night.items()
            for pair in overlapping(
                group,
                *window,
                night=night,
                long_span_hours=view.view.long_span_hours,
            )
        ]

    if args.after_sunset:
        # Needs no night at all: it reads each event's own recorded sunset
        # rather than tonight's, which is what makes it correct on any date.
        selected = after_sunset(selected)

    # Re-sorted after grouping: the batch's rank order is the product, and
    # bucketing by night would otherwise leave the list in night order.
    selected.sort(key=lambda ranked: ranked.rank)

    print(
        render_recommendations(
            selected,
            heading=None,
            verbose=args.verbose,
            limit=None if args.all else (args.limit if args.limit is not None else view.view.limit),
            color=_supports_color(stdout),
            source_urls=view.source_urls,
            show_dates=True,
            reason_limit=view.view.reason_limit,
        ),
        file=stdout,
        end="",
    )
    return 0


#: Flags that select *which events* are shown, and the other flags each one
#: cannot be combined with. Refused rather than ignored: a dropped flag prints a
#: listing that looks like it obeyed.
_INCOMPATIBLE: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    # `--raw` is unranked and unfiltered by definition, and reads events rather
    # than ranked pairs, which is what the filters take. Making it filter is
    # wanted — see the issue — and is not a matter of passing the value through.
    ("raw", "--raw", ("time", "after_sunset", "date", "days", "upcoming", "explain")),
    # `--explain` accounts for one named event, so a filter over the listing has
    # nothing to act on: it would change nothing, or hide the event asked about.
    ("explain", "--explain", ("time", "after_sunset", "date", "days", "upcoming")),
    # `--upcoming` spans many nights, and `overlapping` anchors its window to
    # one. Wanted, and the reason it is refused rather than ignored.
    # `--date` and `--days` also choose *which events* are shown, so a pair
    # would mean silently ignoring one. `--time` and `--after-sunset` narrow
    # what is already chosen, and are honoured (#32).
    ("upcoming", "--upcoming", ("date", "days")),
)

#: Argument names to the flag a person typed, for the message.
_FLAG_NAMES = {
    "time": "--time",
    "after_sunset": "--after-sunset",
    "date": "--date",
    "days": "--days",
    "upcoming": "--upcoming",
    "explain": "--explain",
    "raw": "--raw",
}


def _conflicting_flags(args: argparse.Namespace) -> str | None:
    """The first combination that cannot be honoured, described for a person.

    Checked ahead of every dispatch rather than inside the command that wins, so
    no path can be reached with a flag it is about to discard.
    """
    for attr, flag, incompatible in _INCOMPATIBLE:
        if getattr(args, attr) in (None, False):
            continue
        for other in incompatible:
            if getattr(args, other) not in (None, False):
                return (
                    f"{flag} cannot be combined with {_FLAG_NAMES[other]} — "
                    f"{flag} does not apply it, and ignoring it would print a "
                    "listing that looks like it did"
                )
    return None


def _nights_to_show(args: argparse.Namespace, tonight: date) -> list[date]:
    """Which nights the view covers, in order.

    Defaults to tonight, which is the question `what-do` exists to answer. The
    ranking already spans the whole horizon, so every other night here was
    scored and stored and simply had no way to be asked for (#19).

    Raises:
        ValueError: If the request is unanswerable as written. `--date` and
            `--days` are two ways of naming the same thing, so accepting both
            would mean silently ignoring one.
    """
    if args.date and args.days is not None:
        raise ValueError("--date and --days cannot be combined; they both choose which nights")

    if args.date:
        try:
            return [date.fromisoformat(args.date)]
        except ValueError as exc:
            raise ValueError(f"--date must be YYYY-MM-DD, got {args.date!r}") from exc

    if args.days is not None:
        if args.days < 1:
            raise ValueError(f"--days must be 1 or more, got {args.days}")
        return [tonight + timedelta(days=offset) for offset in range(args.days)]

    return [tonight]


def _cmd_explain(
    selector: str,
    *,
    db_path: Path,
    run_date: date | None,
    load_pairs: PairLoader,
    load_all_events: EventLoader,
    stdout: TextIO,
    stderr: TextIO,
    view: ViewSettings,
) -> int:
    """Account for one event, named by rank or by part of its title.

    Ranked events are searched first, because a rank only means anything there
    and it is what the list actually printed. Falling back to every stored event
    is what lets `--explain` reach the ones `--raw` just made visible: a
    superseded event has no score and no ranking, so it can be named but never
    ranked.
    """
    pairs = load_pairs(
        db_path, run_date=run_date, embedding_model=view.embedding_model
    )
    found = matching(pairs, selector)

    if len(found) == 1:
        print(
            render_explanation(
                found[0].event, found[0].score, found[0].ranking, total=len(pairs)
            ),
            file=stdout,
            end="",
        )
        return 0

    if len(found) > 1:
        # Listed rather than guessed. Picking one silently is how somebody ends
        # up reading the wrong event's explanation and believing it.
        #
        # Folded rather than dumped, on the listing's own terms: show some, say
        # how many there were. A broad selector can match hundreds — `--explain
        # a` matches most of the corpus — and burying the answer under them is
        # no more useful than picking one. The count stays on screen for the
        # same reason it does in the listing: a cut match must not be silent.
        print(f"Error: {selector!r} matches {len(found)} events:", file=stderr)
        for pair in found[: view.view.match_limit]:
            # Handles, not ranks. This list exists to be picked from, so every
            # line must be something you can paste straight back — a rank would
            # invite typing a number that no longer selects anything.
            handle = f"{HANDLE_SIGIL}{short_handle(pair.event.event_id)}"
            print(f"  {handle}  {pair.event.title}", file=stderr)
        if (rest := len(found) - view.view.match_limit) > 0:
            print(f"  + {rest} more match{'' if rest == 1 else 'es'}", file=stderr)
        # Fitted to what was actually typed. Telling somebody who typed `#0f`
        # to "use the #handle" is the same presumption as deciding a bare
        # integer meant a rank — they did use one, they need more of it.
        if selector.startswith(HANDLE_SIGIL):
            print("  Give more of the handle.", file=stderr)
        else:
            print("  Name one with its #handle.", file=stderr)
        return 1

    unranked = _unranked_match(
        selector, load_all_events(db_path, embedding_model=view.embedding_model)
    )
    if unranked is not None:
        print(render_explanation(unranked, None, None, total=len(pairs)), file=stdout, end="")
        return 0

    print(f"Error: no event matches {selector!r}", file=stderr)
    return 1


def _unranked_match(selector: str, events: list[Event]) -> Event | None:
    """An event with no ranking row, named by handle or by part of its title.

    The same two selectors as `matching`, and for the same reason: a handle is
    derived from `event_id`, which every stored event has, so it reaches the
    superseded rows that have no score and no ranking. Those are precisely the
    events `--raw` exists to reveal, and before the handle nothing it showed
    could be named back.
    """
    if selector.startswith(HANDLE_SIGIL):
        prefix = selector[len(HANDLE_SIGIL) :].strip().casefold()
        matches = [e for e in events if short_handle(e.event_id).startswith(prefix)]
    else:
        needle = selector.casefold()
        matches = [e for e in events if e.title and needle in e.title.casefold()]
    return matches[0] if len(matches) == 1 else None


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="what-do", description="What should we do tonight?"
    )
    parser.add_argument("--time", help="Only events overlapping a window, e.g. 20:30-23:30")
    parser.add_argument(
        "--after-sunset", action="store_true", help="Only events starting after sunset"
    )
    parser.add_argument(
        "--raw", action="store_true", help="Every stored event, unranked and unfiltered"
    )
    parser.add_argument(
        "--all", action="store_true", help="Show every ranked event, not just the top ones"
    )
    parser.add_argument(
        "--date", metavar="YYYY-MM-DD", help="Show a night other than tonight"
    )
    parser.add_argument(
        "--upcoming",
        nargs="?",
        type=int,
        const=_UPCOMING_FROM_CONFIG,
        metavar="DAYS",
        help=(
            "One ranked list across the DAYS nights after tonight, best first, "
            "for planning ahead. Defaults to view.upcoming_days in config"
        ),
    )
    parser.add_argument(
        "--days",
        type=int,
        metavar="N",
        help="Show the next N nights, each as its own section",
    )
    parser.add_argument(
        "--explain",
        metavar="EVENT",
        help="Account for one event: its #handle from the list, or part of its title",
    )
    parser.add_argument(
        "--limit",
        type=int,
        metavar="N",
        help="How many events to show (default: view.limit in config)",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Show score components and every reason"
    )
    parser.add_argument("--db", help=f"Path to the database (default: {DEFAULT_DB_PATH})")
    parser.add_argument("--run-date", help="Read an earlier batch, as YYYY-MM-DD")

    subparsers = parser.add_subparsers(dest="command")
    add_source = subparsers.add_parser("add-source", help="Add a handle or venue to seeds.yaml")
    add_source.add_argument("handle", nargs="?", help="Social handle (e.g. @jazzclub)")
    add_source.add_argument("--venue", help="Venue name")
    add_source.add_argument("--address", help="Venue address")
    add_source.add_argument("--seeds-file", help="Path to seeds.yaml (default: data/seeds.yaml)")

    return parser


def run(
    argv: list[str] | None = None,
    *,
    get_now: Callable[[], datetime] = datetime.now,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
    load_pairs: PairLoader = _default_pairs,
    load_all_events: EventLoader = _default_events,
    db_ready: ReadinessCheck = has_schema,
    load_view_settings: ViewSettingsLoader = default_view_settings,
    check_freshness: FreshnessProbe = _default_freshness,
) -> int:
    """Run one CLI invocation and return its exit code.

    Streams and loaders are injected so tests exercise the real argument
    handling without a database or a captured process.

    Args:
        argv: Arguments without the program name. Defaults to `sys.argv[1:]`.
        get_now: Clock, injected. Decides which day "today" means.
        stdout: Where the rendered view goes.
        stderr: Where usage errors go.
        load_pairs: Reads a run's ranked events — the event, its score and
            its placement, joined.
        load_all_events: Reads every stored event, for `--raw`.
        db_ready: Reports whether the database has been initialised. Injected
            with the loaders it guards, so a test that substitutes them does not
            still depend on a real file being on disk.
        load_view_settings: Reads the timezone and rollover hour that decide
            which night is shown. Injected so tests never need a config file.
        check_freshness: Reports whether the forecast behind the ranking has
            aged past the read TTL, or the preference files have moved since
            it was scored. Injected on the same terms as the loaders.

    Returns:
        0 on success, including an empty database. 1 for a usage error.
    """
    out = stdout if stdout is not None else sys.stdout
    err = stderr if stderr is not None else sys.stderr
    args = _build_parser().parse_args(argv)

    # `add-source` deliberately never loads config: it is the one command that
    # works before any batch has run, and giving it a config dependency would
    # add a failure mode it has no use for.
    if args.command == "add-source":
        return _cmd_add_source(args, out, err)

    return _cmd_recommend(
        args,
        get_now=get_now,
        stdout=out,
        stderr=err,
        load_pairs=load_pairs,
        load_all_events=load_all_events,
        db_ready=db_ready,
        view=load_view_settings(),
        check_freshness=check_freshness,
    )


def main() -> None:
    """Entry point for the what-do CLI."""
    sys.exit(run())
