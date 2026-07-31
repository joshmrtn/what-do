"""Handle disambiguation step (batch step 3a).

Classifies probationary candidate_entities as 'venue' or 'person' using an LLM provider,
then evaluates handle promotion.
"""

from __future__ import annotations

import json
import sqlite3
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from src.utils.ollama_client import OllamaClient, OllamaError


class DisambiguationError(Exception):
    """Raised when a handle cannot be classified after retries."""


class DisambiguationProvider(ABC):
    """Classifies a social handle as 'venue' or 'person'."""

    @abstractmethod
    def classify(self, handle: str, context: str) -> Literal["venue", "person"]:
        """Classify a handle given surrounding context text.

        Args:
            handle: The social handle to classify (e.g. '@jazzclub').
            context: Surrounding post caption that mentioned the handle.

        Returns:
            'venue' or 'person'.
        """


_CLASSIFY_PROMPT = """\
You are classifying a social media handle as either a "venue" (a place, business, or \
organisation) or a "person" (an individual human).

Handle: {handle}
Context: {context}

Respond with only valid JSON in this exact format:
{{"classification": "venue"}}
or
{{"classification": "person"}}

Do not include any other text."""

_RETRY_PROMPT = """\
Your previous response was not valid. You must respond with only valid JSON in this \
exact format:
{{"classification": "venue"}}
or
{{"classification": "person"}}

Try again for handle: {handle}"""


class OllamaDisambiguationProvider(DisambiguationProvider):
    """Classifies handles using a local Ollama model.

    Args:
        client: Configured OllamaClient instance.
        model: Ollama model name to use for classification.
    """

    def __init__(self, client: OllamaClient, model: str = "gemma4:e2b") -> None:
        self._client = client
        self._model = model

    def classify(self, handle: str, context: str) -> Literal["venue", "person"]:
        """Classify a handle as 'venue' or 'person' using the LLM.

        Args:
            handle: The social handle to classify.
            context: Surrounding caption text mentioning the handle.

        Returns:
            'venue' or 'person'.

        Raises:
            DisambiguationError: If the model fails to produce a valid response after 1 retry.
        """
        messages = [
            {
                "role": "user",
                "content": _CLASSIFY_PROMPT.format(handle=handle, context=context),
            }
        ]

        raw = self._client.chat(model=self._model, messages=messages)
        result = self._parse(raw)

        if result is None:
            retry_msg = {"role": "user", "content": _RETRY_PROMPT.format(handle=handle)}
            messages = messages + [{"role": "assistant", "content": raw}, retry_msg]
            raw = self._client.chat(model=self._model, messages=messages)
            result = self._parse(raw)

        if result is None:
            raise DisambiguationError(
                f"Failed to classify {handle} after 1 retry. Last response: {raw!r}"
            )

        return result

    def _parse(self, text: str) -> Literal["venue", "person"] | None:
        """Parse a classification response, returning None if invalid."""
        try:
            data = json.loads(text.strip())
            value = data.get("classification", "")
            if value in ("venue", "person"):
                return value  # type: ignore[return-value]
            return None
        except (json.JSONDecodeError, AttributeError):
            return None


class DisambiguationStep:
    """Batch step 3a: classify new probationary handles and update their state."""

    def __init__(
        self,
        db_path: Path,
        provider: DisambiguationProvider,
        logger: Any,
    ) -> None:
        self._db_path = db_path
        self._provider = provider
        self._logger = logger

    def run(self) -> None:
        """Classify all unclassified probationary handles."""
        conn = sqlite3.connect(self._db_path)
        try:
            rows = conn.execute(
                """SELECT id, handle, discovery_context
                   FROM candidate_entities
                   WHERE state = 'probationary' AND llm_classification IS NULL""",
            ).fetchall()

            now = datetime.now(timezone.utc).isoformat()
            for entity_id, handle, context in rows:
                try:
                    classification = self._provider.classify(
                        handle=handle,
                        context=context or "",
                    )
                except Exception as exc:
                    self._logger.error(
                        f"Disambiguation failed for {handle}: {exc}",
                        component="disambiguation",
                        duration_ms=0,
                    )
                    continue

                new_state = "discarded" if classification == "person" else "probationary"
                conn.execute(
                    """UPDATE candidate_entities
                       SET llm_classification = ?, state = ?, updated_at = ?
                       WHERE id = ?""",
                    (classification, new_state, now, entity_id),
                )

            conn.commit()
        finally:
            conn.close()
