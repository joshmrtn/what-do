"""CLI entry points for the what-do application.

Reads precomputed rows and renders them. No network, no LLM, no scoring and no
reordering happen here — the batch already decided, and this is the view onto
that decision.
"""

from __future__ import annotations

import argparse
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Any, Callable, TextIO

import yaml

from src.models.event import Event
from src.presentation.filters import (
    RankedPair,
    after_sunset,
    on_date,
    overlapping,
    parse_time_window,
)
from src.presentation.render import render_raw, render_recommendations
from src.storage.db import DEFAULT_DB_PATH, has_schema
from src.storage.events import load_events
from src.storage.recommendations import load_ranked

_DEFAULT_SEEDS_PATH = Path("data/seeds.yaml")

_NO_DATABASE_MESSAGE = "No database yet — run the overnight batch to build one."
_NO_RECOMMENDATIONS_MESSAGE = "No recommendations yet — run the overnight batch to populate them."

PairLoader = Callable[..., list[RankedPair]]
EventLoader = Callable[..., list[Event]]
ReadinessCheck = Callable[[Path], bool]


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

    if args.handle:
        handle = args.handle if args.handle.startswith("@") else f"@{args.handle}"
        if handle in data.get("handles", []):
            print(f"{handle} is already in seeds.yaml", file=stdout)
            return 0
        data.setdefault("handles", []).append(handle)
        _write_seeds(seeds_path, data)
        print(f"Added {handle} to seeds.yaml", file=stdout)
        return 0

    if args.venue and args.address:
        venues = data.get("venues", [])
        for v in venues:
            if v.get("name") == args.venue:
                print(f"Venue '{args.venue}' is already in seeds.yaml", file=stdout)
                return 0
        venues.append({"name": args.venue, "address": args.address})
        data["venues"] = venues
        _write_seeds(seeds_path, data)
        print(f"Added venue '{args.venue}' to seeds.yaml", file=stdout)
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
) -> int:
    """Render the default view: the latest run, filtered to today."""
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
        print(render_raw(load_all_events(db_path)), file=stdout, end="")
        return 0

    try:
        run_date = date.fromisoformat(args.run_date) if args.run_date else None
    except ValueError:
        print(f"Error: --run-date must be YYYY-MM-DD, got {args.run_date!r}", file=stderr)
        return 1

    try:
        window = parse_time_window(args.time) if args.time else None
    except ValueError as exc:
        print(f"Error: {exc}", file=stderr)
        return 1

    pairs = load_pairs(db_path, run_date=run_date)
    if not pairs:
        print(_NO_RECOMMENDATIONS_MESSAGE, file=stdout)
        return 0

    today = get_now().date()
    # `on_date` drops undated events, so they are re-added rather than filtered:
    # a missing start time is a gap in what we know, not evidence about when.
    view = on_date(pairs, today) + [p for p in pairs if p[1].start_time is None]
    view.sort(key=lambda pair: pair[0].rank)

    if window is not None:
        view = overlapping(view, *window)
    if args.after_sunset:
        view = after_sunset(view)

    print(
        render_recommendations(
            view,
            heading=today.strftime("%A %-d %B"),
            verbose=args.verbose,
            show_all=args.all,
            color=_supports_color(stdout),
        ),
        file=stdout,
        end="",
    )
    return 0


def _supports_color(stream: TextIO) -> bool:
    """ANSI only when the destination is a terminal, so pipes stay clean."""
    return bool(getattr(stream, "isatty", lambda: False)())


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
        "--all", action="store_true", help="Expand the lower-ranked events instead of folding them"
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
    load_pairs: PairLoader = load_ranked,
    load_all_events: EventLoader = load_events,
    db_ready: ReadinessCheck = has_schema,
) -> int:
    """Run one CLI invocation and return its exit code.

    Streams and loaders are injected so tests exercise the real argument
    handling without a database or a captured process.

    Args:
        argv: Arguments without the program name. Defaults to `sys.argv[1:]`.
        get_now: Clock, injected. Decides which day "today" means.
        stdout: Where the rendered view goes.
        stderr: Where usage errors go.
        load_pairs: Reads (recommendation, event) pairs for a run.
        load_all_events: Reads every stored event, for `--raw`.
        db_ready: Reports whether the database has been initialised. Injected
            with the loaders it guards, so a test that substitutes them does not
            still depend on a real file being on disk.

    Returns:
        0 on success, including an empty database. 1 for a usage error.
    """
    out = stdout if stdout is not None else sys.stdout
    err = stderr if stderr is not None else sys.stderr
    args = _build_parser().parse_args(argv)

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
    )


def main() -> None:
    """Entry point for the what-do CLI."""
    sys.exit(run())
