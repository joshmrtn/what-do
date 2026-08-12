"""Handle disambiguation step (batch step 3a).

Classifies probationary candidate_entities as 'venue' or 'person' using an LLM provider,
then evaluates handle promotion.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any, Callable, Literal

from src.config import DEFAULT_DISAMBIGUATION_MODEL
from src.models.candidate_entity import DISCARDED, PROBATIONARY
from src.storage.protocols import EntityRepository
from src.utils.chat_client import ChatClient


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
        client: Any ChatClient (e.g. OllamaClient, GeminiClient).
        model: Model name to use for classification.
    """

    def __init__(self, client: ChatClient, model: str = DEFAULT_DISAMBIGUATION_MODEL) -> None:
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
            if value == "venue":
                return "venue"
            if value == "person":
                return "person"
            return None
        except (json.JSONDecodeError, AttributeError):
            return None


def _default_now() -> datetime:
    return datetime.now(timezone.utc)


class DisambiguationStep:
    """Batch step 3a: classify new probationary handles and update their state."""

    def __init__(
        self,
        entities: EntityRepository,
        provider: DisambiguationProvider,
        logger: Any,
        get_now: Callable[[], datetime] = _default_now,
    ) -> None:
        self._entities = entities
        self._provider = provider
        self._logger = logger
        self._get_now = get_now

    def run(self) -> None:
        """Classify all unclassified probationary handles."""
        now = self._get_now()
        for entity in self._entities.unclassified():
            try:
                classification = self._provider.classify(
                    handle=entity.handle,
                    context=entity.discovery_context or "",
                )
            except Exception as exc:
                self._logger.error(
                    f"Disambiguation failed for {entity.handle}: {exc}",
                    component="disambiguation",
                    duration_ms=0,
                )
                continue

            # A person is left alone; a venue stays probationary until it has
            # earned enough mentions to be promoted.
            state = DISCARDED if classification == "person" else PROBATIONARY
            self._entities.classify(
                entity.entity_id,
                classification=classification,
                state=state,
                now=now,
            )
