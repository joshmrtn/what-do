"""A spinner for the one thing at query time that is not instant.

`what-do` is normally milliseconds. When it refreshes the forecast and re-runs
the pipeline tail it is twenty-odd seconds, and twenty-odd seconds of a blank
terminal reads as a hung command rather than a working one.

Two rules, both about not damaging the thing it decorates:

* **stderr, never stdout.** Stdout is the listing and a reader may pipe it.
* **Only on a terminal.** Piped or redirected, the spinner writes nothing at
  all — carriage returns and escape codes in a captured file are worse than no
  progress indicator.

The ticking runs on a daemon thread. That is not the parallelism `CLAUDE.md`
defers to post-v1: nothing here touches the pipeline, the stages, or storage. It
draws a character and sleeps.
"""

from __future__ import annotations

import itertools
import threading
from contextlib import contextmanager
from typing import Iterator, TextIO

#: Braille dots: one cell wide, so erasing is predictable.
_FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
_INTERVAL_SECONDS = 0.08


def is_interactive(stream: TextIO) -> bool:
    """Whether this stream is a terminal a human is watching."""
    try:
        return bool(stream.isatty())
    except (AttributeError, ValueError):
        # A closed or substituted stream is not a terminal, and asking must
        # never be the thing that fails an invocation.
        return False


@contextmanager
def spinner(message: str, *, stream: TextIO, enabled: bool) -> Iterator[None]:
    """Show a spinner while the body runs, then erase it completely.

    Erasing rather than leaving the line behind is deliberate: the spinner is
    scaffolding, and once the listing is on screen the fact that it took a
    moment is not information the reader still needs.

    Args:
        message: Shown beside the spinner, e.g. "refreshing tonight's forecast".
        stream: Where to draw. Expected to be stderr.
        enabled: False draws nothing, for a pipe or a test.

    Yields:
        None. Whatever the body raises propagates unchanged, with the line
        erased first so a traceback does not land on top of half a spinner.
    """
    if not enabled:
        yield
        return

    stop = threading.Event()
    width = len(message) + 2

    def tick() -> None:
        for frame in itertools.cycle(_FRAMES):
            if stop.is_set():
                return
            stream.write(f"\r{frame} {message}")
            stream.flush()
            stop.wait(_INTERVAL_SECONDS)

    thread = threading.Thread(target=tick, daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join(timeout=1.0)
        stream.write("\r" + " " * width + "\r")
        stream.flush()
