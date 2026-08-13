"""Where bench samples come from: a file you maintain, or the live database.

Sample events are **not** committed. The listings this project reads name real
venues a short walk from the author's home, so the shipped set is anonymised —
invented names over the real shapes — and your own set is gitignored beside
`likes.txt`.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from src.bench.runner import Sample
from src.models.event import Event
from src.storage.protocols import EventRepository

#: Everything a sample may declare. Checked rather than ignored, because a
#: silently dropped `catgeory:` leaves the sample no longer testing the thing it
#: was written for, and nothing says so.
_FIELDS = frozenset(
    {"name", "title", "description", "venue", "location",
     "listing_category", "note", "reference_date"}
)


class SampleError(ValueError):
    """Raised when a sample cannot be read as written."""


def load_samples(path: Path | str) -> list[Sample]:
    """Read a sample file.

    Args:
        path: A YAML file holding a list of samples.

    Returns:
        The samples, in file order.

    Raises:
        SampleError: If a sample is unnamed, unexplained, or names a field that
            does not exist.
    """
    raw = yaml.safe_load(Path(path).read_text()) or []
    if not isinstance(raw, list):
        raise SampleError(f"{path}: expected a list of samples")

    return [_sample(entry, path, index) for index, entry in enumerate(raw, 1)]


def _sample(entry: Any, path: Path | str, index: int) -> Sample:
    """Build one sample, rejecting anything that would silently do nothing."""
    where = f"{path}: sample {index}"
    if not isinstance(entry, dict):
        raise SampleError(f"{where}: expected a mapping")

    unknown = set(entry) - _FIELDS
    if unknown:
        raise SampleError(f"{where}: unknown field(s) {', '.join(sorted(unknown))}")
    if not entry.get("name"):
        raise SampleError(f"{where}: needs a name")
    if not entry.get("note"):
        raise SampleError(
            f"{where} ({entry['name']}): needs a note saying what makes it tricky"
        )

    fields = dict(entry)
    reference_date = fields.pop("reference_date", None)
    return Sample(**fields, reference_date=_parse_date(reference_date, where))


def _parse_date(value: Any, where: str) -> datetime | None:
    """Read a reference date, always aware.

    Naive here would be a bug waiting: the pipeline compares against an aware
    clock, and a naive default that only one path reaches is how the first live
    fetch died.
    """
    if value is None:
        return None
    if isinstance(value, datetime):
        parsed = value
    else:
        try:
            parsed = datetime.fromisoformat(str(value))
        except ValueError as error:
            raise SampleError(f"{where}: bad reference_date {value!r}") from error
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def samples_from_db(repository: EventRepository, event_ids: list[str]) -> list[Sample]:
    """Draw samples from stored events, by id.

    The question worth asking most often is "why did *this* event tag badly",
    and that event is in the database rather than in any fixture. These samples
    are real and stay in memory — never write them to a file the repository
    tracks.

    Takes the repository rather than a path: the bench is a tool, not a second
    storage layer, and reaching for a raw connection here would put the sample
    loader back in the business of knowing what a database is.

    Raises:
        SampleError: If any id is not present, since returning fewer samples
            than asked for would read as a model handling a case it never saw.
    """
    stored = {event.event_id: event for event in repository.load_all()}
    missing = [i for i in event_ids if i not in stored]
    if missing:
        raise SampleError(f"no such event(s): {', '.join(missing)}")

    return [_from_event(stored[i]) for i in event_ids]


def _from_event(event: Event) -> Sample:
    """One stored event as a sample."""
    return Sample(
        name=f"db:{event.event_id[:8]}",
        title=event.title,
        description=event.description,
        venue=event.venue,
        location=event.location,
        listing_category=event.metadata.get("listing_category"),
        note=f"Drawn from the database: {event.event_id}",
    )
