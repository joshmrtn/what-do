"""The read path's logger must never write to stdout.

Stdout is the listing. A structured JSON line in it is not merely ugly — it is
in the stream a reader might pipe, and `get_logger` defaults to stdout, so the
only thing standing between a batch's diagnostics and the top of `what-do` is
this being got right.
"""

from __future__ import annotations

import io
import sys

from src.presentation.cli import _view_logger


def test_the_view_logger_writes_to_stderr_by_default():
    """`get_logger` defaults to stdout, which is where the listing goes.

    Shipped that way once: `reusing 1750 tag vector(s) from earlier runs` landed
    above the first recommendation on every rescore.
    """
    handler = _view_logger()._logger.handlers[0]

    assert handler.stream is sys.stderr


def test_routine_progress_is_not_reported_at_all():
    """A stage's INFO chatter is batch diagnostics, not a message to a reader.

    Suppressed by level rather than by routing, because on a rescore the reader
    asked "what should we do tonight" and every answer to that is on stdout.
    """
    stream = io.StringIO()
    logger = _view_logger(stream=stream)

    logger.info("reusing 1750 tag vector(s) from earlier runs", component="embedding_stage")

    assert stream.getvalue() == ""


def test_something_going_wrong_is_still_reported():
    """A rescore that failed is worth saying, on stderr, without the listing."""
    stream = io.StringIO()
    logger = _view_logger(stream=stream)

    logger.warning("rescore abandoned", component="rescore")

    assert "rescore abandoned" in stream.getvalue()


def test_an_error_is_reported():
    stream = io.StringIO()
    logger = _view_logger(stream=stream)

    logger.error("weather fetch failed", component="enrichment")

    assert "weather fetch failed" in stream.getvalue()
