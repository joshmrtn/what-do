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

from src.utils.chat_client import ChatClient


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
        tags: Descriptive tags for the event (minimum min_tags).
        summary: One-sentence event summary.
    """

    title: str | None
    venue: str | None
    start_time: datetime | None
    end_time: datetime | None
    tags: list[str]
    summary: str


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
  "tags": ["tag1", "tag2", "tag3", "tag4", "tag5"],
  "summary": "One sentence describing the event."
}}

Rules:
- tags must contain at least {min_tags} descriptive labels (genre, activity type, atmosphere, etc.)
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
  "tags": ["tag1", "tag2", ...],
  "summary": "One sentence."
}}

Remember: at least {min_tags} tags required. Output ONLY the JSON."""


class OllamaExtractionProvider(ExtractionProvider):
    """Extracts structured event data using a local Ollama LLM.

    Args:
        client: Any ChatClient (e.g. OllamaClient, GeminiClient).
        model: Model name (default gemma4:e4b).
        min_tags: Minimum number of tags required in the output.
    """

    def __init__(self, client: ChatClient, model: str = "gemma4:e4b", min_tags: int = 5) -> None:
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
            text=text, min_tags=self._min_tags, date_context=date_context
        )
        messages = [{"role": "user", "content": prompt}]
        chat_kwargs: dict[str, Any] = {"model": self._model, "messages": messages}
        if image_bytes is not None:
            chat_kwargs["images"] = [image_bytes]

        raw = self._client.chat(**chat_kwargs)
        result, error = self._parse_and_validate(raw)

        if result is None:
            retry_prompt = _RETRY_PROMPT.format(reason=error, min_tags=self._min_tags)
            retry_messages = messages + [
                {"role": "assistant", "content": raw},
                {"role": "user", "content": retry_prompt},
            ]
            retry_kwargs: dict[str, Any] = {"model": self._model, "messages": retry_messages}
            if image_bytes is not None:
                retry_kwargs["images"] = [image_bytes]

            raw = self._client.chat(**retry_kwargs)
            result, error = self._parse_and_validate(raw)

        if result is None:
            raise ExtractionError(f"Extraction failed after 1 retry: {error}")

        return result

    def _parse_and_validate(self, text: str) -> tuple[ExtractionResult | None, str]:
        """Parse raw LLM output into ExtractionResult, returning (result, error_reason)."""
        try:
            data = json.loads(text.strip())
        except json.JSONDecodeError:
            return None, "JSON parse error — response was not valid JSON"

        tags = data.get("tags")
        if not isinstance(tags, list) or len(tags) < self._min_tags:
            count = len(tags) if isinstance(tags, list) else 0
            return None, f"tag count {count} is below minimum {self._min_tags}"

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
        ), ""

    @staticmethod
    def _parse_dt(value: object) -> datetime | None:
        """Parse an ISO 8601 string into a datetime, returning None on failure."""
        if not value or not isinstance(value, str):
            return None
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            return None
