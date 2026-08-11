"""The `integration` tier: a reachability probe and a skip nobody can miss.

`integration` tests use a real Ollama for embeddings. On a machine without one
— a fresh clone, CI — they cannot run, and that is a legitimate state rather
than a failure. What is *not* acceptable is reporting green having skipped
them: that is the same defect as a deselected tier, where the summary says
everything passed and the cross-module coverage was never exercised.

So the skip is loud. It names Ollama and the host that was tried, and it prints
a summary banner that survives `-q`.

This lives in an importable module rather than in `conftest.py` so that the
tests covering it can load the real plugin instead of a copy of its logic.
"""

from __future__ import annotations

import os

import pytest
import requests

DEFAULT_OLLAMA_HOST = "http://localhost:11434"

# Short: the probe runs once and a wrong answer is cheap to retry, while a long
# timeout is paid on every run on a machine that has no Ollama at all.
PROBE_TIMEOUT = 2.0

_SKIP_REASON = "Ollama unreachable at {host}"
_WARNING = (
    "integration tier did not run: Ollama unreachable at {host}. "
    "Cross-module coverage was NOT checked."
)

# Resolved on first use and reused for the rest of the session. Probing per test
# costs a connection attempt each time, which is what turns an unreachable host
# from instant into slow.
_probe_result: bool | None = None


def ollama_host() -> str:
    """The host the probe and the production clients both read."""
    return os.environ.get("OLLAMA_HOST", DEFAULT_OLLAMA_HOST).rstrip("/")


def ollama_available(host: str, timeout: float = PROBE_TIMEOUT) -> bool:
    """Whether something answers Ollama's model listing at `host`.

    Checks `/api/tags` rather than opening a socket: a port that accepts a
    connection is not necessarily an Ollama, and the tier needs the service.
    """
    try:
        response = requests.get(f"{host}/api/tags", timeout=timeout)
    except requests.RequestException:
        return False
    return response.status_code == 200


def probe_cache() -> bool | None:
    """The cached probe result, or None if it has not been resolved yet."""
    return _probe_result


def set_probe_cache(value: bool | None) -> None:
    """Force the cached result. For tests that must restore prior state."""
    global _probe_result
    _probe_result = value


def reset_probe_cache() -> None:
    """Drop the cached result so the next check probes again."""
    set_probe_cache(None)


def _reachable() -> bool:
    global _probe_result
    if _probe_result is None:
        _probe_result = ollama_available(ollama_host())
    return _probe_result


def pytest_runtest_setup(item: pytest.Item) -> None:
    """Skip an `integration` test when Ollama is not there to answer it."""
    if "integration" in item.keywords and not _reachable():
        pytest.skip(_SKIP_REASON.format(host=ollama_host()))


def pytest_terminal_summary(terminalreporter: pytest.TerminalReporter) -> None:
    """Announce an unrun tier in the summary, where `-q` cannot hide it."""
    if _probe_result is False:
        terminalreporter.write_sep("=", "INCOMPLETE COVERAGE", red=True, bold=True)
        terminalreporter.write_line(_WARNING.format(host=ollama_host()))
