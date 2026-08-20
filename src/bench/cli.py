"""`what-do-bench` — run samples past models and print what came back.

Its own small composition root. Nothing in the pipeline imports the bench and
the bench imports no pipeline stage, only the providers, so the rule that only a
composition root names an implementation still holds.

The bench never gates anything. It exits non-zero only when it could not run at
all — never because a model gave an answer someone disliked.
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

import requests

from src.bench.disambiguation import (
    HandleSample,
    HandleVariant,
    format_classifications,
    run_handle_variant,
)
from src.bench.report import format_report, load_run, write_run
from src.bench.runner import Measurement, Variant, run_variant
from src.bench.samples import SampleError, load_samples, samples_from_db
from src.composition.storage import build_view_storage
from src.config import (
    DEFAULT_EMBEDDING_MODEL,
    NetworkConfig,
    NetworkPolicy,
    Patience,
)
from src.network.policy import RequestPolicy
from src.network.throttle import InMemoryThrottle
from src.storage.sqlite.connection import DEFAULT_DB_PATH
from src.utils.chat_client import GENERATION_PATIENCE
from src.utils.ollama_client import OllamaClient

DEFAULT_SAMPLES = Path("data/bench-samples.yaml")
DEFAULT_LOG_DIR = Path("logs")

#: Generous for the same reason the pipeline's is: a single extraction against a
#: local model on CPU runs for minutes, and a bench that times out measures the
#: timeout.
_TIMEOUT_SECONDS = 3600

#: The bench's own policy, because the bench is its own composition root and its
#: numbers are deliberately not the pipeline's.
#:
#: It waits far longer — measuring a model that may be slow is the whole point,
#: and a ceiling that ended the run would be reporting our patience rather than
#: its speed. And it makes **one attempt**: a silent retry would fold two runs
#: into one measurement, which is worse than a failed sample.
#:
#: Declared here rather than read from `config.yaml` so the bench still runs
#: without one, and so a deliberate difference is visible where it is used.
_BENCH_POLICY = "bench_local_model"

#: The handles the disambiguation prompt has actually been wrong about. Held
#: here rather than in the sample file because they are invented already — no
#: real handle appears — and because deleting `tests/model/` must not lose them.
_HANDLE_SAMPLES = [
    HandleSample(
        name="obvious-venue",
        handle="@thevaultlounge",
        context="Come enjoy live jazz at @thevaultlounge this Saturday — doors at 7pm!",
        note="A handle naming a room, with the room's own listing around it. "
        "The easy case, and the one that was a compliance test.",
    ),
    HandleSample(
        name="person-who-sounds-like-a-place",
        handle="@thebrasstap",
        context="@thebrasstap is back on the decks Friday — third time this month.",
        note="A DJ whose stage name is a bar. Only the context says so, which "
        "is the whole reason the classifier gets context at all.",
    ),
]


class BenchError(ValueError):
    """Raised when the bench cannot be set up as asked."""


def parse_args(argv: list[str]) -> argparse.Namespace:
    """Parse the bench's arguments."""
    parser = argparse.ArgumentParser(
        prog="what-do-bench",
        description="Run samples past one or more models and print what came back.",
    )
    parser.add_argument("command", choices=["extraction", "disambiguation"])
    parser.add_argument(
        "--variant",
        action="append",
        default=[],
        metavar="[NAME=]MODEL",
        help="A model to compare. Repeat it. Name it separately when comparing "
        "two variants of one model.",
    )
    parser.add_argument("--samples", default=str(DEFAULT_SAMPLES))
    parser.add_argument(
        "--from-db",
        default="",
        metavar="ID,ID",
        help="Draw samples from stored events instead of a file.",
    )
    parser.add_argument("--db", default=str(DEFAULT_DB_PATH))
    parser.add_argument("--host", default="http://localhost:11434")
    parser.add_argument(
        "--baseline",
        help="A previously recorded run to mark changes against.",
    )
    args = parser.parse_args(argv)
    args.from_db = [i for i in args.from_db.split(",") if i]
    return args


def _bench_policy(host: str) -> RequestPolicy:
    """One policy over the model under test, at the bench's own numbers."""
    waiting = Patience(
        timeout_seconds=float(_TIMEOUT_SECONDS),
        max_attempts=1,
        backoff_base_seconds=0.0,
        backoff_max_seconds=0.0,
    )
    network = NetworkConfig(
        # The host policy and the generation patience carry the same numbers
        # here: everything the bench asks of this host is a model call.
        policies={
            _BENCH_POLICY: NetworkPolicy(
                min_interval_seconds=0.0,
                timeout_seconds=waiting.timeout_seconds,
                max_attempts=waiting.max_attempts,
                backoff_base_seconds=waiting.backoff_base_seconds,
                backoff_max_seconds=waiting.backoff_max_seconds,
                cache_ttl=None,
            )
        },
        hosts={urlsplit(host).hostname or host: _BENCH_POLICY},
        patience={GENERATION_PATIENCE: waiting},
    )
    return RequestPolicy(
        network=network,
        throttle=InMemoryThrottle(get_now=_utc_now, sleep=time.sleep),
        sleep=time.sleep,
        random=random.random,
    )


def _utc_now() -> datetime:
    """The bench's clock. Its own, because it is its own composition root."""
    return datetime.now(timezone.utc)


def build_variants(specs: list[str], host: str) -> list[Variant]:
    """Turn `--variant` strings into variants.

    Accepts `model` or `name=model`. The second form matters because the
    comparison worth making most often is two variants of the *same* model —
    one prompt against another, one input shape against another — and they need
    telling apart in the table.
    """
    if not specs:
        raise BenchError("give at least one --variant")

    policy = _bench_policy(host)
    variants: list[Variant] = []
    seen: set[str] = set()
    for spec in specs:
        name, _, model = spec.partition("=")
        if not model:
            name, model = spec, spec
        if name in seen:
            raise BenchError(f"variant {name!r} given twice")
        seen.add(name)
        variants.append(
            Variant(
                name=name,
                model=model,
                client=OllamaClient(
                    host,
                    session=requests.Session(),
                    policy=policy,
                    get_now=_utc_now,
                ),
            )
        )
    return variants


def _run_extraction(args: argparse.Namespace) -> str:
    """Every sample past every variant, in sample order."""
    if args.from_db:
        # Through the one storage factory, like both other roots, rather
        # than naming an implementation a second time.
        storage = build_view_storage(args.db, DEFAULT_EMBEDDING_MODEL)
        samples = samples_from_db(storage.events, args.from_db)
    else:
        samples = load_samples(args.samples)

    variants = build_variants(args.variant, host=args.host)
    measurements = [
        run_variant(sample, variant) for sample in samples for variant in variants
    ]

    baseline = load_run(args.baseline) if args.baseline else None
    _record(measurements)
    return format_report(samples, measurements, baseline=baseline)


def _run_disambiguation(args: argparse.Namespace) -> str:
    """The handle classifier, over the handles it has been wrong about."""
    specs = build_variants(args.variant, host=args.host)
    variants = [
        HandleVariant(name=v.name, model=v.model, client=v.client) for v in specs
    ]
    results = [
        run_handle_variant(sample, variant)
        for sample in _HANDLE_SAMPLES
        for variant in variants
    ]
    return format_classifications(_HANDLE_SAMPLES, results)


def _record(measurements: list[Measurement]) -> None:
    """Write the run beside the batch logs, so the next run can diff it."""
    DEFAULT_LOG_DIR.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    path = DEFAULT_LOG_DIR / f"bench-{stamp}.json"
    write_run(measurements, path)
    print(f"recorded: {path}", file=sys.stderr)


def run() -> int:
    """Entry point for `what-do-bench`."""
    args = parse_args(sys.argv[1:])
    try:
        if args.command == "extraction":
            print(_run_extraction(args))
        else:
            print(_run_disambiguation(args))
    except (BenchError, SampleError) as exc:
        print(f"bench: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
