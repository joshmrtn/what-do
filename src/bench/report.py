"""Print a bench run for a person, and keep it so the next run can be compared.

Nothing here judges. A model answering `jazz` where another said `music` is
different, not wrong, and a bench that scored that would teach us to prefer
whichever model best matches a guess we wrote down in advance. The output is a
table; the reading is the person's job.

What it *can* do is spare the person half the work. Comparing two runs by eye is
the tedious part, so a run is written to a file and the next one can mark what
moved.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.bench.runner import Measurement, Sample
from src.models.tag import Tag

_SUMMARY_WIDTH = 60


def format_report(
    samples: list[Sample],
    measurements: list[Measurement],
    baseline: list[Measurement] | None = None,
) -> str:
    """Render one run, optionally marking what moved since a previous one.

    Args:
        samples: The samples run, in the order they should be read.
        measurements: What each variant returned.
        baseline: A previous run to compare against, by (sample, variant).

    Returns:
        The report, ready to print.
    """
    previous = {(m.sample, m.variant): m for m in baseline or []}
    by_sample: dict[str, list[Measurement]] = {}
    for measurement in measurements:
        by_sample.setdefault(measurement.sample, []).append(measurement)

    lines: list[str] = []
    for sample in samples:
        rows = by_sample.get(sample.name, [])
        if not rows:
            continue
        lines.append(f"\n{sample.name}")
        if sample.note:
            lines.append(f"  {' '.join(sample.note.split())}")
        width = max(len(row.variant) for row in rows)
        for row in rows:
            lines.extend(_row(row, width, previous.get((row.sample, row.variant))))
    return "\n".join(lines)


def _row(measurement: Measurement, width: int, before: Measurement | None) -> list[str]:
    """One variant's answer, and its second line when it has more to say."""
    head = f"  {measurement.variant.ljust(width)}  {measurement.seconds:>7.1f}s  "
    if measurement.error:
        return [f"{head}{measurement.error}"]

    lines = [head + _tags(measurement.tags, before)]
    detail = []
    if measurement.summary:
        detail.append(_clip(measurement.summary))
    if measurement.degradation:
        detail.append(f"[{measurement.degradation}]")
    if detail:
        lines.append(f"  {' ' * width}           {'  '.join(detail)}")
    return lines


def _tags(tags: list[Tag], before: Measurement | None) -> str:
    """Tags with weights, marked against a baseline when there is one.

    A variant the baseline never saw is *new*, not changed — marking every tag
    as added would bury the one comparison the baseline exists for.
    """
    rendered = [f"{tag.text}({tag.weight})" for tag in tags]
    if before is None:
        return " ".join(rendered) or "—"

    now = {tag.text for tag in tags}
    then = {tag.text for tag in before.tags}
    marked = [
        f"+{tag.text}({tag.weight})" if tag.text not in then else f"{tag.text}({tag.weight})"
        for tag in tags
    ]
    marked.extend(f"-{text}" for text in sorted(then - now))
    return " ".join(marked) or "—"


def _clip(text: str) -> str:
    """One line of summary, since the table is read in a terminal."""
    collapsed = " ".join(text.split())
    if len(collapsed) <= _SUMMARY_WIDTH:
        return collapsed
    return collapsed[: _SUMMARY_WIDTH - 1] + "…"


def write_run(measurements: list[Measurement], path: Path | str) -> None:
    """Record a run so a later one can be compared against it."""
    payload = [
        {
            "sample": m.sample,
            "variant": m.variant,
            "tags": [[t.text, t.weight] for t in m.tags],
            "summary": m.summary,
            "degradation": m.degradation,
            "seconds": m.seconds,
            "error": m.error,
        }
        for m in measurements
    ]
    Path(path).write_text(json.dumps(payload, indent=1))


def load_run(path: Path | str) -> list[Measurement]:
    """Read a recorded run back."""
    raw = json.loads(Path(path).read_text())
    return [
        Measurement(
            sample=entry["sample"],
            variant=entry["variant"],
            tags=[Tag(text=text, weight=weight) for text, weight in entry["tags"]],
            summary=entry["summary"],
            degradation=entry["degradation"],
            seconds=entry["seconds"],
            error=entry["error"],
        )
        for entry in raw
    ]
