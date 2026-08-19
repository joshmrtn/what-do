"""Structured JSON logger built on top of Python's stdlib logging."""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from typing import IO

from src.utils.secret import scrub


class _JSONFormatter(logging.Formatter):
    """Formats log records as single-line JSON with required structured fields.

    Every log line in the process is rendered here, which is why this is where
    credentials are scrubbed: a caller cannot opt out, and a logger built
    somewhere that has no composition root to ask for a redactor is covered
    anyway. See `src/utils/secret.py` for why the registry is process-wide.
    """

    def format(self, record: logging.LogRecord) -> str:
        return json.dumps({
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "component": getattr(record, "component", ""),
            "severity": record.levelname,
            "duration_ms": getattr(record, "duration_ms", 0),
            "message": scrub(record.getMessage()),
        })


class StructuredLogger:
    """Thin wrapper around logging.Logger with a structured call signature."""

    def __init__(self, logger: logging.Logger) -> None:
        self._logger = logger

    def _emit(self, level: int, message: str, component: str, duration_ms: int) -> None:
        self._logger.log(
            level,
            message,
            extra={"component": component, "duration_ms": duration_ms},
        )

    def debug(self, message: str, *, component: str = "", duration_ms: int = 0) -> None:
        """Emit a DEBUG entry."""
        self._emit(logging.DEBUG, message, component, duration_ms)

    def info(self, message: str, *, component: str = "", duration_ms: int = 0) -> None:
        """Emit an INFO entry."""
        self._emit(logging.INFO, message, component, duration_ms)

    def warning(self, message: str, *, component: str = "", duration_ms: int = 0) -> None:
        """Emit a WARNING entry."""
        self._emit(logging.WARNING, message, component, duration_ms)

    def error(self, message: str, *, component: str = "", duration_ms: int = 0) -> None:
        """Emit an ERROR entry."""
        self._emit(logging.ERROR, message, component, duration_ms)


def get_logger(
    name: str,
    *,
    stream: IO[str] | None = None,
    level: str = "INFO",
) -> StructuredLogger:
    """Create a structured JSON logger.

    Args:
        name: Logger name (used by stdlib logging internally).
        stream: Output stream. Defaults to stdout.
        level: Log level string (DEBUG, INFO, WARNING, ERROR).

    Returns:
        StructuredLogger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    logger.handlers.clear()
    logger.propagate = False

    handler = logging.StreamHandler(stream or sys.stdout)
    handler.setFormatter(_JSONFormatter())
    logger.addHandler(handler)

    return StructuredLogger(logger)
