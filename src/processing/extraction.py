"""LLM-based event extraction — Pass 1.

Converts raw event text (title + description) into structured data:
tags, summary, and optional title/venue/time corrections.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from src.config import DEFAULT_EXTRACTION_MODEL, SETTINGS
from src.models.tag import DEFAULT_WEIGHT, Tag, clamp_weight
from src.utils.chat_client import ChatClient
from src.utils.text import normalize_embedding_text


class ExtractionError(Exception):
    """Raised when structured extraction fails after retries."""


@dataclass
class ExtractionResult:
    """Structured output from LLM Pass 1.

    Fields:
        title: Extracted or corrected event title (None if not determinable).
        venue: Extracted venue name (None if not determinable).
        start_time: Parsed start datetime (None if not determinable).
        end_time: Parsed end datetime (None if not determinable).
        tags: Weighted descriptive tags for the event (minimum min_tags).
        summary: One-sentence event summary.
        setting: "indoor", "outdoor", or "unknown".
    """

    title: str | None
    venue: str | None
    start_time: datetime | None
    end_time: datetime | None
    tags: list[Tag]
    summary: str
    setting: str = "unknown"


def _decode_json_object(text: str) -> dict[str, Any] | None:
    """Decode the JSON object in a model reply, tolerating a markdown fence.

    Models wrap JSON in ```json fences however firmly the prompt says not to —
    measured, gemma4:e4b did it to roughly 40% of replies, and each one cost a
    whole event despite the JSON inside being complete and correct. Constrained
    decoding stops it at the source; this stops it for any model that slips
    through, including one reached through a different provider.

    Falls back to the span between the first `{` and last `}` rather than
    matching fences specifically, which also absorbs a stray "Here you go:".
    Returns None when nothing object-shaped is present — prose stays a failure.
    """
    candidates = [text.strip()]
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        candidates.append(text[start : end + 1])

    for candidate in candidates:
        try:
            decoded = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(decoded, dict):
            return decoded
    return None


#: Tags that name the *idea* of a tag rather than anything about an event.
#: Measured coming out of both gemma and Gemini on name-only listings; they are
#: embedded and scored against the user's preferences like any real tag.
_PLACEHOLDER_TAGS = frozenset(
    {"genre", "music genre", "musical genre", "artist", "event", "performance type"}
)


def _drop_meaningless(tags: list[Tag], source_text: str) -> list[Tag]:
    """Remove tags that carry no information about the event.

    Two kinds. A **placeholder** names the idea of a tag — `genre`, `artist` —
    and says nothing. A **title echo** repeats the act's own name back:
    `Lee Hawkins` produced the tag `lee hawkins`, which then gets embedded and
    compared against preference lines as though it meant something.

    The echo test is deliberately exact and anchored to the title line only, so
    `Jazz Night` keeps its `jazz` tag. A looser substring rule would delete
    exactly the genre tags that make a title worth reading.
    """
    title = source_text.split("\n", 1)[0].strip().casefold()
    return [
        tag
        for tag in tags
        if tag.text.casefold() not in _PLACEHOLDER_TAGS
        and tag.text.casefold() != title
    ]


def _parse_setting(raw: Any) -> str:
    """Coerce the model's `setting` to the allowed enum.

    Anything unrecognised becomes "unknown" rather than failing the extraction —
    a bad enum value is not worth a retry costing minutes of local LLM time, and
    "unknown" is a safe verdict that simply earns no weather adjustment.
    """
    if isinstance(raw, str) and raw.strip().lower() in SETTINGS:
        return raw.strip().lower()
    return "unknown"


class ExtractionProvider(ABC):
    """Extracts structured data from raw event text."""

    @abstractmethod
    def extract(
        self,
        text: str,
        image_bytes: bytes | None = None,
        reference_date: datetime | None = None,
    ) -> ExtractionResult:
        """Extract structured event data from text.

        Args:
            text: Raw event text (title + description combined).
            image_bytes: Optional raw image bytes to pass to a multimodal model.
            reference_date: Optional "today" anchor so the model can resolve
                relative dates (e.g. "this Saturday") to absolute dates.

        Returns:
            ExtractionResult with extracted fields.

        Raises:
            ExtractionError: If extraction fails after retries.
        """


_EXTRACT_PROMPT = """\
{date_context}Extract structured event information from the text below and respond with ONLY valid JSON.

Text:
{text}

Required JSON format:
{{
  "title": "event title or null",
  "venue": "venue name or null",
  "start_time": "ISO 8601 datetime or null (e.g. 2026-06-22T20:00:00)",
  "end_time": "ISO 8601 datetime or null",
  "tags": [{{"tag": "short lowercase phrase", "weight": 0.0-1.0}}],
  "summary": "One sentence describing the event.",
  "setting": "indoor" | "outdoor" | "unknown"
}}

Rules:
- List every tag the text supports — no more, no fewer. There is no minimum and
  no target number. A terse listing yields few tags; a detailed one yields more.
  Never pad the list to reach a count.
- A genre or style tag is allowed ONLY when the text NAMES it. The title counts
  as text: "Blues Brunch" names blues, "A Fleetwood Mac Tribute" names Fleetwood
  Mac. A performer's proper name does NOT name a genre — a person's or band's
  name tells you nothing about their style, and a word that merely appears
  inside a name is not evidence of anything. If the text does not name the
  style, emit no style tag.
- Do not emit the performing act's own name as a tag — that describes who, not
  what. An act the event is ABOUT rather than performed by, such as the subject
  of a tribute, may be tagged.
- Do not emit placeholder tags such as "genre", "music genre", "artist" or
  "event" — a tag must carry meaning on its own.
- "Event category" is a section heading from a listing site. It tells you the
  broad kind of event and nothing more specific.
- "weight" is how CENTRAL the tag is to what the event actually IS:
  1.0 = the main activity or defining feature; the reason someone attends
  0.5 = a real but secondary attribute
  0.1 = incidental context (the kind of venue, the day of week, decor)
- Weights must discriminate. Do not give every tag a similar weight.
- Judge centrality from the event text, not from what is typical of the venue type.
- "setting" is where the event is physically held: "outdoor" only if it takes
  place outside. A venue that merely has a patio is "indoor". If the text does
  not make it clear, use "unknown" rather than guessing.
- summary must be exactly one sentence
- Use null (not empty string) for unknown fields
- Output ONLY the JSON object — no explanation, no markdown"""

_RETRY_PROMPT = """\
Your previous response was invalid: {reason}

You must respond with ONLY valid JSON matching this exact format:
{{
  "title": "event title or null",
  "venue": "venue name or null",
  "start_time": "ISO 8601 datetime or null",
  "end_time": "ISO 8601 datetime or null",
  "tags": [{{"tag": "short phrase", "weight": 0.0-1.0}}, ...],
  "summary": "One sentence.",
  "setting": "indoor" | "outdoor" | "unknown"
}}

Remember: tag only what the text supports, each with a centrality weight where
1.0 is the event's defining feature and 0.1 is incidental context.
Output ONLY the JSON."""


class OllamaExtractionProvider(ExtractionProvider):
    """Extracts structured event data using a local Ollama LLM.

    Args:
        client: Any ChatClient (e.g. OllamaClient, GeminiClient).
        model: Model name (default gemma4:e4b).
        min_tags: How few tags a reply may carry before it counts as failed.
            One by default: a terse listing honestly supports one tag, and a
            higher floor is what made models pad rather than answer.
    """

    def __init__(
        self,
        client: ChatClient,
        model: str = DEFAULT_EXTRACTION_MODEL,
        min_tags: int = 1,
    ) -> None:
        self._client = client
        self._model = model
        self._min_tags = min_tags

    def extract(
        self,
        text: str,
        image_bytes: bytes | None = None,
        reference_date: datetime | None = None,
    ) -> ExtractionResult:
        """Extract structured event data from text, with one retry on schema failure.

        Args:
            text: Raw event text.
            image_bytes: Optional raw image bytes for multimodal extraction.
            reference_date: Optional "today" anchor for resolving relative dates.

        Returns:
            Validated ExtractionResult.

        Raises:
            ExtractionError: If both attempts produce invalid output.
        """
        date_context = ""
        if reference_date is not None:
            date_context = (
                f"Today's date is {reference_date.date().isoformat()}. "
                "Resolve any relative dates (e.g. 'this Saturday', 'next Thursday') "
                "to absolute ISO 8601 dates against it.\n\n"
            )
        prompt = _EXTRACT_PROMPT.format(
            text=text, date_context=date_context
        )
        messages = [{"role": "user", "content": prompt}]
        chat_kwargs: dict[str, Any] = {"model": self._model, "messages": messages}
        if image_bytes is not None:
            chat_kwargs["images"] = [image_bytes]

        raw = self._client.chat(**chat_kwargs)
        result, error = self._parse_and_validate(raw, text)

        if result is None:
            retry_prompt = _RETRY_PROMPT.format(reason=error)
            retry_messages = messages + [
                {"role": "assistant", "content": raw},
                {"role": "user", "content": retry_prompt},
            ]
            retry_kwargs: dict[str, Any] = {"model": self._model, "messages": retry_messages}
            if image_bytes is not None:
                retry_kwargs["images"] = [image_bytes]

            raw = self._client.chat(**retry_kwargs)
            result, error = self._parse_and_validate(raw, text)

        if result is None:
            raise ExtractionError(f"Extraction failed after 1 retry: {error}")

        return result

    def _parse_and_validate(
        self, reply: str, source_text: str
    ) -> tuple[ExtractionResult | None, str]:
        """Parse raw LLM output into ExtractionResult, returning (result, error_reason).

        Args:
            reply: What the model sent back.
            source_text: The event text the model was given, needed to tell a
                tag that describes the event from one that merely echoes its
                title back.
        """
        data = _decode_json_object(reply)
        if data is None:
            return None, "JSON parse error — response was not valid JSON"

        raw_tags = data.get("tags")
        if not isinstance(raw_tags, list):
            return None, f"tag count 0 is below minimum {self._min_tags}"

        tags = _drop_meaningless(self._parse_tags(raw_tags), source_text)
        if len(tags) < self._min_tags:
            return None, f"tag count {len(tags)} is below minimum {self._min_tags}"

        summary = data.get("summary")
        if not summary or not isinstance(summary, str):
            return None, "summary field is missing or not a string"

        start_time = self._parse_dt(data.get("start_time"))
        end_time = self._parse_dt(data.get("end_time"))

        return ExtractionResult(
            title=data.get("title") or None,
            venue=data.get("venue") or None,
            start_time=start_time,
            end_time=end_time,
            tags=tags,
            summary=summary,
            setting=_parse_setting(data.get("setting")),
        ), ""

    @staticmethod
    def _parse_tags(raw_tags: list[Any]) -> list[Tag]:
        """Build weighted tags from model output, tolerating schema drift.

        Accepts both {"tag": ..., "weight": ...} objects and bare strings.
        Entries with no usable text are dropped rather than failing the whole
        extraction; weights are clamped into range.

        Tag text is normalised here rather than at embedding time so the stored
        tag, its vector, and anything displayed to the user all agree.
        """
        tags: list[Tag] = []
        for entry in raw_tags:
            if isinstance(entry, str):
                text, weight = entry, DEFAULT_WEIGHT
            elif isinstance(entry, dict):
                raw_text = entry.get("tag", entry.get("text", ""))
                text = raw_text if isinstance(raw_text, str) else ""
                weight = clamp_weight(entry.get("weight"))
            else:
                continue
            normalized = normalize_embedding_text(text)
            if normalized:
                tags.append(Tag(text=normalized, weight=weight))
        return tags

    @staticmethod
    def _parse_dt(value: object) -> datetime | None:
        """Parse an ISO 8601 string into a datetime, returning None on failure."""
        if not value or not isinstance(value, str):
            return None
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            return None
