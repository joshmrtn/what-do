"""Put one sample in front of one variant and record what came back.

The bench answers a question the test suite deliberately cannot: is this model,
prompt or input shape any *good*. That is why nothing here asserts on tag
content — `jazz` where another model said `music` is different, not wrong — and
why the output is a table for a person rather than a pass or a fail.

Its one hard requirement is that the prompt comes from the production path. A
bench that builds its own prompt measures the bench.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable

from src.models.event import Event
from src.models.tag import Tag
from src.processing.extraction import OllamaExtractionProvider
from src.processing.extraction_input import extraction_input
from src.utils.chat_client import ChatClient, LLMError

#: Samples describe an event, not a database row, so the fields an Event needs
#: only for storage are filled with something inert and identical everywhere.
_EPOCH = datetime(2026, 1, 1, tzinfo=timezone.utc)


@dataclass(frozen=True)
class Sample:
    """One event to put in front of a model.

    `note` is the point of the sample and not decoration: a bench sample earns
    its place by being *tricky*, and the next reader needs to know what the
    trick was. Samples are drawn from listings that actually broke — anonymised,
    because the real ones name venues a short walk from the author's home.
    """

    name: str
    title: str | None = None
    description: str | None = None
    venue: str | None = None
    location: str | None = None
    listing_category: str | None = None
    note: str | None = None
    reference_date: datetime | None = None

    def as_event(self) -> Event:
        """The Event `extraction_input` expects.

        Built rather than faked so the bench feeds the production input builder
        exactly what the pipeline would. A sample shaped like anything else
        would diverge from production silently, which is the failure the bench
        is supposed to detect in *models*, not commit itself.
        """
        metadata = {}
        if self.listing_category:
            metadata["listing_category"] = self.listing_category
        return Event(
            event_id=f"bench:{self.name}",
            source_event_candidates=[],
            source_type="bench",
            created_at=_EPOCH,
            updated_at=_EPOCH,
            title=self.title,
            description=self.description,
            venue=self.venue,
            location=self.location,
            metadata=metadata,
        )


@dataclass(frozen=True)
class Variant:
    """One thing to compare: a model, a prompt, or an input shape.

    The issue framed this as a model, which is the narrower half. What is
    actually wanted most often is an A/B of a *change* — the venue-line question
    could not be asked of a bench that varied only the model, and that was the
    question we had.
    """

    name: str
    model: str
    client: ChatClient
    #: Defaults to the production builder, so a variant that does not say
    #: otherwise measures what the pipeline really sends.
    input_builder: Callable[[Event], str] = extraction_input
    min_tags: int = 1


@dataclass(frozen=True)
class Measurement:
    """What one variant did with one sample. No verdict, by design."""

    sample: str
    variant: str
    tags: list[Tag] = field(default_factory=list)
    summary: str | None = None
    degradation: str | None = None
    seconds: float = 0.0
    error: str | None = None


def run_variant(sample: Sample, variant: Variant) -> Measurement:
    """Run one sample through one variant, recording rather than judging.

    An unreachable model is recorded and returned, never raised: the bench
    exists to compare, and a table missing a column is worth less than one whose
    column reads `unreachable`.
    """
    provider = OllamaExtractionProvider(
        client=variant.client, model=variant.model, min_tags=variant.min_tags
    )
    text = variant.input_builder(sample.as_event())
    started = time.monotonic()

    try:
        result = provider.extract(text, reference_date=sample.reference_date)
    except LLMError as exc:
        return Measurement(
            sample=sample.name,
            variant=variant.name,
            seconds=round(time.monotonic() - started, 1),
            error=f"{type(exc).__name__}: {exc}",
        )

    return Measurement(
        sample=sample.name,
        variant=variant.name,
        tags=result.tags,
        summary=result.summary,
        degradation=result.degradation,
        seconds=round(time.monotonic() - started, 1),
    )
