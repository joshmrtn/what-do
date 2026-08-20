"""Thin HTTP client for the Ollama API.

**Locality is an exemption from spacing, not from retry.** There is no third
party at `localhost` to be polite to, so its policy sets no interval — but a
dropped connection or a model still loading is our own pipeline's problem, and
an extraction that fails at minute three has cost minutes. So this goes through
the same `RequestPolicy` as every other caller, and the difference between a
local model and a hosted one is a set of numbers in config rather than a branch
here (#36).

The host is whatever `OLLAMA_HOST` names. Its policy is looked up by that
address, so pointing it at another machine picks that machine's manners up
rather than quietly keeping an exemption written for this one.
"""

from __future__ import annotations

import base64
import time
from datetime import datetime
from typing import Any, Callable
from urllib.parse import urlsplit

import requests

from src.network.http import requests_transient_check
from src.network.policy import RequestPolicy
from src.network.protocols import NullCache
from src.utils.chat_client import GENERATION_PATIENCE, LLMError
from src.utils.llm_transcript import TranscriptSink

#: Why neither endpoint caches. A prompt is not a cacheable resource, and both
#: callers already avoid the repeat call one layer up rather than replaying its
#: answer: extraction skips on `extraction_input_hash`, and a tag vector is
#: stored once per `(tag text, model)` by the caller that knows what identifies it.
_NO_CACHE = (
    "a prompt is not a cacheable resource; both callers skip the repeat call one "
    "layer up rather than replaying an answer"
)


class OllamaError(LLMError):
    """Raised when an Ollama API call fails."""


class OllamaClient:
    """Wraps the Ollama /api/chat and /api/embed endpoints.

    Args:
        host: Base URL of the Ollama server (e.g. 'http://localhost:11434').
        session: Transport, injected so the policy is the only way out of the
            process and a test never reaches the network.
        policy: The shared request policy. Required and keyword-only: an optional
            dependency is one that arrives nowhere.
        get_now: Injected clock, for reading a `Retry-After` given as a date.
        transcript: Optional sink recording every call verbatim. None disables
            recording entirely, which is what a normal run wants — the file is
            a debugging aid, not part of the batch's output.
        component: Name recorded against this client's calls, so extraction and
            disambiguation stay distinguishable in one transcript.
        options: Sampling and runtime options (temperature, top_p, num_ctx, ...)
            sent with every chat request. None sends none, leaving the model's
            own defaults in force.
        think: Whether a thinking-capable model should reason before answering.
            None omits the field; False is a real instruction and is sent.
        response_format: Ollama's `format` — "json" constrains decoding to valid
            JSON. None omits it.
        keep_alive: How long the server keeps this model resident after a call
            ("30m", "0" to unload immediately). None omits it and leaves the
            server default in force — which on this host never expires, so a
            model loaded once squats its whole footprint forever.

    Raises:
        OllamaError: If `host` names no hostname, since the policy is per host.
    """

    def __init__(
        self,
        host: str,
        *,
        session: requests.Session,
        policy: RequestPolicy,
        get_now: Callable[[], datetime],
        transcript: TranscriptSink | None = None,
        component: str = "ollama",
        options: dict[str, Any] | None = None,
        think: bool | None = None,
        response_format: str | dict[str, Any] | None = None,
        keep_alive: str | None = None,
    ) -> None:
        self._host = host.rstrip("/")
        hostname = urlsplit(self._host).hostname
        if hostname is None:
            raise OllamaError(
                f"Ollama host {host!r} names no host, so no policy can be found "
                "for it. Give OLLAMA_HOST a full URL, e.g. http://localhost:11434"
            )
        self._hostname = hostname
        self._session = session
        self._policy = policy
        # Resolved now rather than on the first call: the batch builds its
        # clients before it fetches anything, so an unassigned model host is
        # named at 02:00:01 instead of after the ingestion it would waste.
        policy.limits_for(hostname)
        self._is_transient = requests_transient_check(get_now=get_now)
        self._transcript = transcript
        self._component = component
        self._options = options
        self._think = think
        self._response_format = response_format
        self._keep_alive = keep_alive
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
            ConfigError: If the host has no policy, or the generation patience
                is not declared.
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
        # Each is omitted unless configured, so an unconfigured client posts
        # exactly what it always did. `think` is tested against None rather
        # than falsiness: False is the value we most want to send.
        if self._response_format is not None:
            payload["format"] = self._response_format
        if self._think is not None:
            payload["think"] = self._think
        if self._options:
            payload["options"] = self._options
        if self._keep_alive is not None:
            payload["keep_alive"] = self._keep_alive

        # Named by the method rather than by the constructor, because the shape
        # of the request is what it is: generating tokens takes minutes, and it
        # takes them whoever is running the model.
        body = self._post("/api/chat", payload, model, patience=GENERATION_PATIENCE)
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
        embed_payload: dict[str, Any] = {"model": model, "input": text}
        if self._keep_alive is not None:
            embed_payload["keep_alive"] = self._keep_alive

        body = self._post(
            "/api/embed",
            embed_payload,
            model,
            # No patience: an embedding is the ordinary ask of this host, and it
            # is measured in milliseconds. The read path embeds edited preference
            # lines, where a generation's timeout would be a hang.
            patience=None,
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
        *,
        patience: str | None,
        transcript_request: dict[str, Any] | None = None,
        transcript_response: Any = None,
    ) -> dict[str, Any]:
        """POST to Ollama through the policy, recording every attempt.

        The recording happens inside the attempt rather than at each call site so
        no failure path can quietly skip it — a call that timed out is exactly
        the one worth having a record of, and a retried call is two exchanges
        rather than one.

        The transport's own errors are raised untouched from inside, so the
        policy's check can read a status off them, and converted to `OllamaError`
        outside. Converting first would hand the check our own type, which
        carries no status, and every 503 would be judged permanent — retry
        configured, wired, and dead.
        """
        recorded_request = transcript_request if transcript_request is not None else payload

        def attempt(timeout: float) -> dict[str, Any]:
            self._sequence += 1
            sequence = self._sequence
            started = time.monotonic()

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
                    duration_ms=int((time.monotonic() - started) * 1000),
                    error=error,
                )

            try:
                resp = self._session.post(
                    f"{self._host}{path}", json=payload, timeout=timeout
                )
            except requests.Timeout as exc:
                record(None, None, f"timed out: {exc}")
                raise
            except requests.ConnectionError as exc:
                record(None, None, f"refused: {exc}")
                raise

            if resp.status_code != 200:
                record(
                    None,
                    resp.status_code,
                    f"HTTP {resp.status_code}: {resp.text[:200]}",
                )
                resp.raise_for_status()
                raise OllamaError(
                    f"HTTP {resp.status_code}: {resp.text[:200]}"
                )

            body: dict[str, Any] = resp.json()
            record(
                transcript_response(body) if transcript_response is not None else body,
                resp.status_code,
                None,
            )
            return body

        try:
            return self._policy.call(
                host=self._hostname,
                perform=attempt,
                is_transient=self._is_transient,
                cache=NullCache(reason=_NO_CACHE),
                label=self._component,
                patience=patience,
            )
        except OllamaError:
            raise
        except requests.Timeout as exc:
            raise OllamaError(f"timed out: {exc}") from exc
        except requests.ConnectionError as exc:
            raise OllamaError(f"refused: {exc}") from exc
        except requests.HTTPError as exc:
            response = exc.response
            status = "unknown" if response is None else response.status_code
            said = "" if response is None else response.text[:200]
            raise OllamaError(f"HTTP {status}: {said}") from exc
