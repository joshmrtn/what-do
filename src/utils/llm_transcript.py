"""Verbatim record of every model call, for debugging what the LLM actually saw.

A structured log line says an extraction failed. It does not say what we asked
or what came back, and on CPU inference a single call costs minutes — far too
much to rediscover by re-running variations. This writes the whole exchange to
a sidecar JSON Lines file instead, one object per call, so a failure can be
diagnosed from the run that already happened.

Kept out of the main batch log deliberately: full prompts would bury the
per-source ingestion lines that log is actually read for.
"""

from __future__ import annotations

import copy
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Protocol

#: Marker replacing base64 image data, which is enormous and tells us nothing
#: that its size does not.
IMAGES_KEY = "images"


class TranscriptSink(Protocol):
    """Anything a client can hand a model call to.

    Exists so `OllamaClient` depends on the shape rather than the file-writing
    class, and tests can substitute a recorder that keeps calls in memory.
    """

    def record(
        self,
        *,
        component: str,
        model: str,
        sequence: int,
        request: dict[str, Any],
        response: dict[str, Any] | None,
        status: int | None,
        duration_ms: int,
        error: str | None = None,
    ) -> None:
        """Append one call to the transcript."""
        ...


def _redact_images(request: dict[str, Any]) -> dict[str, Any]:
    """Copy a request payload with image data reduced to its encoded length.

    Deep-copies first: the payload handed in is the one about to go over the
    wire, and redacting it in place would send the marker to the model.
    """
    redacted = copy.deepcopy(request)
    messages = redacted.get("messages")
    if not isinstance(messages, list):
        return redacted

    for message in messages:
        if not isinstance(message, dict):
            continue
        images = message.get(IMAGES_KEY)
        if isinstance(images, list):
            message[IMAGES_KEY] = [
                {"bytes": len(img) if isinstance(img, (str, bytes)) else 0}
                for img in images
            ]
    return redacted


class LLMTranscript:
    """Appends one JSON object per model call to a JSON Lines file.

    Args:
        path: File to append to. Parent directories are created.
        get_now: Injectable clock supplying each record's timestamp.
    """

    def __init__(self, path: Path | str, get_now: Callable[[], datetime]) -> None:
        self._path = Path(path)
        self._get_now = get_now
        self._path.parent.mkdir(parents=True, exist_ok=True)
        # Line buffered, so a run that is still going can be watched, and a run
        # that dies mid-call still leaves everything up to that call on disk.
        self._handle = self._path.open("a", buffering=1, encoding="utf-8")

    def record(
        self,
        *,
        component: str,
        model: str,
        sequence: int,
        request: dict[str, Any],
        response: dict[str, Any] | None,
        status: int | None,
        duration_ms: int,
        error: str | None = None,
    ) -> None:
        """Append one call to the transcript.

        Args:
            component: Which caller made the call (e.g. "extraction").
            model: Model name as sent.
            sequence: This call's ordinal within the client's lifetime. Not an
                attempt number — a client cannot tell a retry from a fresh call,
                so a retried extraction simply shows up as two adjacent records.
            request: The complete payload as sent, images excepted.
            response: The complete decoded response body, or None if none arrived.
            status: HTTP status, or None if the request never completed.
            duration_ms: Wall-clock time for the call.
            error: Failure description, omitted from the record when absent.
        """
        entry: dict[str, Any] = {
            "timestamp": self._get_now().isoformat(),
            "component": component,
            "model": model,
            "sequence": sequence,
            "status": status,
            "duration_ms": duration_ms,
            "request": _redact_images(request),
            "response": response,
        }
        if error is not None:
            entry["error"] = error

        # default=str rather than a raise: a transcript is a debugging aid and
        # must never be the reason a twelve-hour batch dies.
        self._handle.write(json.dumps(entry, default=str) + "\n")

    def close(self) -> None:
        """Close the underlying file."""
        if not self._handle.closed:
            self._handle.close()

    def __enter__(self) -> LLMTranscript:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
