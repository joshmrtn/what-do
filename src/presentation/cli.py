"""CLI entry points for the what-do application."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

_DEFAULT_SEEDS_PATH = Path("data/seeds.yaml")


def _load_seeds_raw(path: Path) -> dict:
    if not path.exists():
        return {"handles": [], "venues": []}
    with open(path) as f:
        return yaml.safe_load(f) or {"handles": [], "venues": []}


def _write_seeds(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True)


def _cmd_add_source(args: argparse.Namespace) -> int:
    seeds_path = Path(args.seeds_file) if args.seeds_file else _DEFAULT_SEEDS_PATH
    data = _load_seeds_raw(seeds_path)

    if args.handle:
        handle = args.handle if args.handle.startswith("@") else f"@{args.handle}"
        if handle in data.get("handles", []):
            print(f"{handle} is already in seeds.yaml")
            return 0
        data.setdefault("handles", []).append(handle)
        _write_seeds(seeds_path, data)
        print(f"Added {handle} to seeds.yaml")
        return 0

    if args.venue and args.address:
        venues = data.get("venues", [])
        for v in venues:
            if v.get("name") == args.venue:
                print(f"Venue '{args.venue}' is already in seeds.yaml")
                return 0
        venues.append({"name": args.venue, "address": args.address})
        data["venues"] = venues
        _write_seeds(seeds_path, data)
        print(f"Added venue '{args.venue}' to seeds.yaml")
        return 0

    print("Error: provide --handle or both --venue and --address", file=sys.stderr)
    return 1


def main() -> None:
    """Entry point for the what-do CLI."""
    parser = argparse.ArgumentParser(prog="what-do")
    subparsers = parser.add_subparsers(dest="command")

    add_source = subparsers.add_parser("add-source", help="Add a handle or venue to seeds.yaml")
    add_source.add_argument("handle", nargs="?", help="Social handle (e.g. @jazzclub)")
    add_source.add_argument("--venue", help="Venue name")
    add_source.add_argument("--address", help="Venue address")
    add_source.add_argument("--seeds-file", help="Path to seeds.yaml (default: data/seeds.yaml)")

    args = parser.parse_args()

    if args.command == "add-source":
        sys.exit(_cmd_add_source(args))
    else:
        print("what-do: no command specified. Try 'what-do add-source --help'")
        sys.exit(0)
