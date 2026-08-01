"""Thin HTTP client for the Ollama API."""

from __future__ import annotations

import base64
from typing import Any

import requests

from src.utils.chat_client import LLMError


class OllamaError(LLMError):
    """Raised when an Ollama API call fails."""


class OllamaClient:
    """Wraps the Ollama /api/chat endpoint.

    Args:
        host: Base URL of the Ollama server (e.g. 'http://localhost:11434').
        timeout: Request timeout in seconds.
    """

    def __init__(self, host: str, timeout: int = 60) -> None:
        self._host = host.rstrip("/")
        self._timeout = timeout

    def chat(
        self,
        model: str,
        messages: list[dict[str, Any]],
        images: list[bytes] | None = None,
    ) -> str:
        """Send a chat request and return the assistant message content.

        Args:
            model: Ollama model name (e.g. 'gemma4:e4b').
            messages: List of message dicts with 'role' and 'content'.
            images: Optional raw image bytes to attach to the last user message.

        Returns:
            The assistant's reply as a string.

        Raises:
            OllamaError: On HTTP error, timeout, or connection failure.
        """
        payload_messages = [dict(m) for m in messages]

        if images:
            encoded = [base64.b64encode(img).decode("ascii") for img in images]
            payload_messages[-1]["images"] = encoded

        payload = {
            "model": model,
            "messages": payload_messages,
            "stream": False,
        }

        try:
            resp = requests.post(
                f"{self._host}/api/chat",
                json=payload,
                timeout=self._timeout,
            )
        except requests.Timeout as exc:
            raise OllamaError(f"timed out: {exc}") from exc
        except requests.ConnectionError as exc:
            raise OllamaError(f"refused: {exc}") from exc

        if resp.status_code != 200:
            raise OllamaError(f"HTTP {resp.status_code}: {resp.text[:200]}")

        return str(resp.json()["message"]["content"])

    def embed(self, model: str, text: str) -> list[float]:
        """Generate an embedding vector for a single piece of text.

        Args:
            model: Embedding model name (e.g. 'nomic-embed-text').
            text: Text to embed.

        Returns:
            The embedding as a list of floats.

        Raises:
            OllamaError: On HTTP error, timeout, connection failure, or a
                response body that does not carry an embedding.
        """
        try:
            resp = requests.post(
                f"{self._host}/api/embed",
                json={"model": model, "input": text},
                timeout=self._timeout,
            )
        except requests.Timeout as exc:
            raise OllamaError(f"timed out: {exc}") from exc
        except requests.ConnectionError as exc:
            raise OllamaError(f"refused: {exc}") from exc

        if resp.status_code != 200:
            raise OllamaError(f"HTTP {resp.status_code}: {resp.text[:200]}")

        embeddings = resp.json().get("embeddings")
        if not isinstance(embeddings, list) or not embeddings:
            raise OllamaError(
                f"unexpected response shape from /api/embed for model {model!r}"
            )

        return [float(x) for x in embeddings[0]]
