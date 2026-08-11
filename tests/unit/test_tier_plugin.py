"""Tests for the `integration` tier's reachability probe and its loud skip.

The subject under test is pytest's own reporting behaviour — whether a skip
reports as a skip, names its cause, and survives `-q`. Asserting that directly
needs a real pytest run, so these drive throwaway suites through the bundled
`pytester` fixture.

The suites load `tests.tier_plugin` itself rather than a copy of its logic, so
what runs here is the code that runs in production. Reachability is steered by
pointing `OLLAMA_HOST` at a closed port or at a stub server bound to localhost:
no service is mocked, and neither case touches a network.
"""

from __future__ import annotations

from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, HTTPServer
import threading

import pytest

from tests import tier_plugin

# A port nothing listens on: connecting refuses immediately rather than waiting
# out a timeout, which keeps the unreachable case as fast as the reachable one.
CLOSED_HOST = "http://127.0.0.1:1"

_SUITE = """
import pytest

@pytest.mark.integration
def test_marked():
    pass

def test_unmarked():
    pass
"""

_INI = """
[pytest]
markers =
    integration: real SQLite and real Ollama embeddings
"""


@pytest.fixture(autouse=True)
def _restore_probe_cache() -> Iterator[None]:
    """Keep a sub-run's probe result from leaking into the real session.

    The plugin module is shared with the session running these tests, so a
    forced-unreachable sub-run would otherwise poison the cache and skip the
    real suite's integration tests.
    """
    previous = tier_plugin.probe_cache()
    tier_plugin.reset_probe_cache()
    yield
    tier_plugin.set_probe_cache(previous)


@pytest.fixture
def stub_ollama() -> Iterator[str]:
    """Serve 200 on /api/tags from an ephemeral localhost port."""

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 - name fixed by BaseHTTPRequestHandler
            self.send_response(200 if self.path == "/api/tags" else 404)
            self.end_headers()
            self.wfile.write(b'{"models": []}')

        def log_message(self, *args: object) -> None:
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()


def _run(
    pytester: pytest.Pytester,
    monkeypatch: pytest.MonkeyPatch,
    host: str,
    *args: str,
) -> pytest.RunResult:
    pytester.makeini(_INI)
    pytester.makepyfile(_SUITE)
    pytester.makeconftest('pytest_plugins = ["tests.tier_plugin"]')
    monkeypatch.setenv("OLLAMA_HOST", host)
    return pytester.runpytest(*args)


def test_integration_test_skips_when_ollama_is_unreachable(
    pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = _run(pytester, monkeypatch, CLOSED_HOST)

    result.assert_outcomes(skipped=1, passed=1, failed=0, errors=0)


def test_skip_reason_names_ollama_and_the_host(
    pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = _run(pytester, monkeypatch, CLOSED_HOST, "-rs")

    result.stdout.fnmatch_lines(["*SKIPPED*Ollama*127.0.0.1:1*"])


def test_the_warning_survives_quiet_mode(
    pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A quiet run must not reduce the warning to an `s` in the progress dots."""

    result = _run(pytester, monkeypatch, CLOSED_HOST, "-q")

    result.stdout.fnmatch_lines(["*integration tier did not run*Ollama*"])


def test_integration_test_runs_when_ollama_is_reachable(
    pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch, stub_ollama: str
) -> None:
    result = _run(pytester, monkeypatch, stub_ollama)

    result.assert_outcomes(passed=2, skipped=0)


def test_no_warning_when_ollama_is_reachable(
    pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch, stub_ollama: str
) -> None:
    result = _run(pytester, monkeypatch, stub_ollama)

    assert "integration tier did not run" not in result.stdout.str()


def test_unmarked_tests_run_when_ollama_is_unreachable(
    pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The probe gates one tier, not the session."""

    result = _run(pytester, monkeypatch, CLOSED_HOST, "-v")

    result.stdout.fnmatch_lines(["*test_unmarked*PASSED*"])


def test_a_directory_named_integration_does_not_imply_the_marker(
    pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`item.keywords` carries every parent's name, not only markers.

    Our real suite keeps these tests in `tests/integration/`, so a keyword
    check skips the whole directory — including the nine unmarked smoke tests
    that need no Ollama at all.
    """
    pytester.makeini(_INI)
    pytester.makeconftest('pytest_plugins = ["tests.tier_plugin"]')
    pytester.mkpydir("integration").joinpath("test_inside.py").write_text(
        "def test_unmarked_but_in_the_directory():\n    pass\n"
    )
    monkeypatch.setenv("OLLAMA_HOST", CLOSED_HOST)

    result = pytester.runpytest("-v")

    result.assert_outcomes(passed=1, skipped=0)


def test_probe_is_resolved_once_per_session(
    pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Three integration tests, one probe.

    Probing per test costs a connection attempt each time, which is what makes
    an unreachable host slow rather than instant.
    """
    calls: list[str] = []

    def counting_probe(host: str, timeout: float = tier_plugin.PROBE_TIMEOUT) -> bool:
        calls.append(host)
        return False

    monkeypatch.setattr(tier_plugin, "ollama_available", counting_probe)

    pytester.makeini(_INI)
    pytester.makeconftest('pytest_plugins = ["tests.tier_plugin"]')
    pytester.makepyfile(
        """
        import pytest

        @pytest.mark.integration
        def test_one(): pass

        @pytest.mark.integration
        def test_two(): pass

        @pytest.mark.integration
        def test_three(): pass
        """
    )
    result = pytester.runpytest()

    result.assert_outcomes(skipped=3)
    assert len(calls) == 1


def test_probe_reports_false_for_a_closed_port() -> None:
    assert tier_plugin.ollama_available("http://127.0.0.1:1") is False


def test_probe_reports_true_for_a_served_tags_endpoint(stub_ollama: str) -> None:
    assert tier_plugin.ollama_available(stub_ollama) is True


def test_probe_reports_false_when_the_endpoint_errors(stub_ollama: str) -> None:
    """A host that answers but not as Ollama is not a usable Ollama."""

    assert tier_plugin.ollama_available(f"{stub_ollama}/wrong-prefix") is False
