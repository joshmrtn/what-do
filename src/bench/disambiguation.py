"""The disambiguation half of the bench.

Kept separate from extraction rather than forced through one type. The two
providers answer different questions — extraction returns weighted tags and a
summary, disambiguation returns one word — and squeezing `venue` into a field
named `summary` would make the recorded runs lie about what they hold.

Everything else is the same discipline: the production provider builds the
prompt, an unreachable model is recorded rather than raised, and nothing here
decides whether an answer is right.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from src.ingestion.disambiguation import (
    DisambiguationError,
    OllamaDisambiguationProvider,
)
from src.utils.chat_client import ChatClient, LLMError


@dataclass(frozen=True)
class HandleSample:
    """One handle, in the context that is supposed to explain it."""

    name: str
    handle: str
    context: str
    note: str


@dataclass(frozen=True)
class HandleVariant:
    """One model to ask."""

    name: str
    model: str
    client: ChatClient


@dataclass(frozen=True)
class Classification:
    """What one variant made of one handle. No verdict, by design."""

    sample: str
    variant: str
    answer: str | None = None
    seconds: float = 0.0
    error: str | None = None


def run_handle_variant(sample: HandleSample, variant: HandleVariant) -> Classification:
    """Ask one variant about one handle, recording rather than judging."""
    provider = OllamaDisambiguationProvider(client=variant.client, model=variant.model)
    started = time.monotonic()

    try:
        answer = provider.classify(handle=sample.handle, context=sample.context)
    # `DisambiguationError` too, and it is the more interesting of the two:
    # a model that ignores the output contract is exactly what a bench is
    # for, and raising would lose every other variant's answer with it.
    except (LLMError, DisambiguationError) as exc:
        return Classification(
            sample=sample.name,
            variant=variant.name,
            seconds=round(time.monotonic() - started, 1),
            error=f"{type(exc).__name__}: {exc}",
        )

    return Classification(
        sample=sample.name,
        variant=variant.name,
        answer=answer,
        seconds=round(time.monotonic() - started, 1),
    )


def format_classifications(
    samples: list[HandleSample], results: list[Classification]
) -> str:
    """Render a disambiguation run for a person to read."""
    by_sample: dict[str, list[Classification]] = {}
    for result in results:
        by_sample.setdefault(result.sample, []).append(result)

    lines: list[str] = []
    for sample in samples:
        rows = by_sample.get(sample.name, [])
        if not rows:
            continue
        lines.append(f"\n{sample.name}  {sample.handle}")
        lines.append(f"  {' '.join(sample.note.split())}")
        width = max(len(row.variant) for row in rows)
        for row in rows:
            answer = row.error or row.answer or "—"
            lines.append(f"  {row.variant.ljust(width)}  {row.seconds:>7.1f}s  {answer}")
    return "\n".join(lines)
