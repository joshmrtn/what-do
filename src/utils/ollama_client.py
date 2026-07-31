"""Thin HTTP client for the Ollama API."""

from __future__ import annotations

import base64

import requests


class OllamaError(Exception):
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
        messages: list[dict],
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

        return resp.json()["message"]["content"]
