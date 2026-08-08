"""Thin HTTP client for the Ollama API."""

from __future__ import annotations

import base64
import time
from typing import Any

import requests

from src.utils.chat_client import LLMError
from src.utils.llm_transcript import TranscriptSink


class OllamaError(LLMError):
    """Raised when an Ollama API call fails."""


class OllamaClient:
    """Wraps the Ollama /api/chat endpoint.

    Args:
        host: Base URL of the Ollama server (e.g. 'http://localhost:11434').
        timeout: Request timeout in seconds.
        transcript: Optional sink recording every call verbatim. None disables
            recording entirely, which is what a normal run wants — the file is
            a debugging aid, not part of the batch's output.
        component: Name recorded against this client's calls, so extraction and
            disambiguation stay distinguishable in one transcript.
    """

    def __init__(
        self,
        host: str,
        timeout: int = 60,
        transcript: TranscriptSink | None = None,
        component: str = "ollama",
    ) -> None:
        self._host = host.rstrip("/")
        self._timeout = timeout
        self._transcript = transcript
        self._component = component
        self._sequence = 0

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

        payload: dict[str, Any] = {
            "model": model,
            "messages": payload_messages,
            "stream": False,
        }

        body = self._post("/api/chat", payload, model)
        return str(body["message"]["content"])

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
        body = self._post(
            "/api/embed",
            {"model": model, "input": text},
            model,
            # A 768-float vector per event would bury the transcript, and the
            # thing worth knowing about an embedding call is what went in.
            transcript_request={"model": model, "input_length": len(text)},
            transcript_response=lambda b: {
                "embedding_count": len(b.get("embeddings") or []),
            },
        )

        embeddings = body.get("embeddings")
        if not isinstance(embeddings, list) or not embeddings:
            raise OllamaError(
                f"unexpected response shape from /api/embed for model {model!r}"
            )

        return [float(x) for x in embeddings[0]]

    def _post(
        self,
        path: str,
        payload: dict[str, Any],
        model: str,
        transcript_request: dict[str, Any] | None = None,
        transcript_response: Any = None,
    ) -> dict[str, Any]:
        """POST to Ollama, recording the exchange before any error escapes.

        The recording happens here rather than at each call site so no failure
        path can quietly skip it — a call that timed out is exactly the one
        worth having a record of.
        """
        self._sequence += 1
        sequence = self._sequence
        recorded_request = transcript_request if transcript_request is not None else payload
        started = time.monotonic()

        def elapsed_ms() -> int:
            return int((time.monotonic() - started) * 1000)

        def record(
            response: dict[str, Any] | None, status: int | None, error: str | None
        ) -> None:
            if self._transcript is None:
                return
            self._transcript.record(
                component=self._component,
                model=model,
                sequence=sequence,
                request=recorded_request,
                response=response,
                status=status,
                duration_ms=elapsed_ms(),
                error=error,
            )

        try:
            resp = requests.post(f"{self._host}{path}", json=payload, timeout=self._timeout)
        except requests.Timeout as exc:
            message = f"timed out: {exc}"
            record(None, None, message)
            raise OllamaError(message) from exc
        except requests.ConnectionError as exc:
            message = f"refused: {exc}"
            record(None, None, message)
            raise OllamaError(message) from exc

        if resp.status_code != 200:
            message = f"HTTP {resp.status_code}: {resp.text[:200]}"
            record(None, resp.status_code, message)
            raise OllamaError(message)

        body: dict[str, Any] = resp.json()
        record(
            transcript_response(body) if transcript_response is not None else body,
            resp.status_code,
            None,
        )
        return body
