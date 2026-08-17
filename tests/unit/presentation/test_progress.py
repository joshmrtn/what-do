"""Tests for the query-time spinner.

Its whole job is to be invisible when it should be and harmless when it is not.
"""

from __future__ import annotations

import io

import pytest

from src.presentation.progress import is_interactive, spinner


class _NotATerminal(io.StringIO):
    def isatty(self) -> bool:
        return False


class _ATerminal(io.StringIO):
    def isatty(self) -> bool:
        return True


class TestIsInteractive:
    def test_a_terminal_is_interactive(self):
        assert is_interactive(_ATerminal()) is True

    def test_a_pipe_is_not(self):
        assert is_interactive(_NotATerminal()) is False

    def test_a_closed_stream_is_not_an_error(self):
        """Asking must never be the thing that fails an invocation.

        A closed stream raises `ValueError` from `isatty`, and a listing lost to
        a progress indicator would be an absurd way to fail.
        """
        stream = io.StringIO()
        stream.close()

        assert is_interactive(stream) is False


class TestSpinner:
    def test_it_writes_nothing_when_disabled(self):
        """A piped listing must not collect carriage returns."""
        stream = io.StringIO()

        with spinner("working", stream=stream, enabled=False):
            pass

        assert stream.getvalue() == ""

    def test_it_erases_itself(self):
        """The spinner is scaffolding, not part of the output.

        Whatever it drew, the last thing it does is blank the line and return
        the cursor — so nothing it wrote is still on screen afterwards.
        """
        stream = io.StringIO()

        with spinner("working", stream=stream, enabled=True):
            pass

        written = stream.getvalue()
        assert written.endswith("\r")
        assert written.endswith("\r" + " " * (len("working") + 2) + "\r")

    def test_the_body_still_runs(self):
        ran = []

        with spinner("working", stream=io.StringIO(), enabled=True):
            ran.append(True)

        assert ran == [True]

    def test_an_exception_propagates_and_the_line_is_cleared(self):
        """A traceback must not land on top of half a spinner."""
        stream = io.StringIO()

        with pytest.raises(ValueError):
            with spinner("working", stream=stream, enabled=True):
                raise ValueError("boom")

        assert stream.getvalue().endswith("\r")

    def test_nothing_reaches_stdout(self, capsys):
        """Belt and braces: the stream is a parameter, and it is the only outlet."""
        with spinner("working", stream=io.StringIO(), enabled=True):
            pass

        assert capsys.readouterr().out == ""
