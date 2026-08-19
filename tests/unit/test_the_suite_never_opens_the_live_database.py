"""No test may open `database/event_hub.db`. Enforced, not remembered.

Thirty-five tests in `test_cli.py` were opening the developer's real 52MB
database on every run, and had been for as long as the rescore seam existed.
The harness injected four collaborators and left three on their production
defaults, each of which builds repositories against `DEFAULT_DB_PATH`.

Nothing caught it because nothing *could*: a test that reads production and
happens to get an answer it likes is indistinguishable from one that is
isolated. It surfaced only when a new column reached `_SCHEMA` and the live
database — which no migration had touched yet — started answering `no such
column` to thirty-five unrelated rendering tests.

Three things make this worth a permanent guard rather than a one-off fix:

- **The suite's verdict depended on machine state.** A fresh clone has no
  database and takes a different path; a machine mid-migration fails.
- **The read path writes.** `rescore_if_stale` reaches `preference_revisions
  .record` and `rescores.record` when it judges a listing stale, so a unit test
  was one injected clock away from writing to production.
- **It is invisible by construction.** The next seam added to `run()` without a
  test double reintroduces it silently, exactly as this one did.
"""

from __future__ import annotations

import textwrap

_OPENS_THE_LIVE_DATABASE = """
    import sqlite3
    from src.storage.sqlite.connection import DEFAULT_DB_PATH

    def test_reaches_production():
        sqlite3.connect(DEFAULT_DB_PATH)
"""

_OPENS_ITS_OWN_DATABASE = """
    import sqlite3

    def test_uses_its_own(tmp_path):
        sqlite3.connect(tmp_path / "scratch.db")
"""

_SWALLOWS_EVERY_EXCEPTION = """
    import sqlite3
    from src.storage.sqlite.connection import DEFAULT_DB_PATH

    def test_swallows_everything():
        try:
            sqlite3.connect(DEFAULT_DB_PATH)
        except Exception:
            pass
"""


def _run(pytester, body: str):
    """Run one generated test under the real guard.

    The plugin is loaded by name rather than copied, so this exercises what
    `addopts` actually registers instead of a second implementation of it.
    """
    pytester.makepyfile(textwrap.dedent(body))
    return pytester.runpytest("-p", "tests.live_db_guard", "-p", "no:cacheprovider")


class TestTheGuard:
    def test_opening_the_live_database_fails_the_test(self, pytester):
        result = _run(pytester, _OPENS_THE_LIVE_DATABASE)

        result.assert_outcomes(failed=1)

    def test_the_failure_names_the_database_and_says_what_to_do(self, pytester):
        """A guard that fires without explaining itself costs an hour.

        Whoever trips this is usually adding a seam to `run()`, and the fix is
        to inject a double for it — not to point the test at another file.
        """
        result = _run(pytester, _OPENS_THE_LIVE_DATABASE)

        output = result.stdout.str()
        assert "event_hub.db" in output
        assert "inject" in output.lower()

    def test_a_test_with_its_own_database_is_untouched(self, pytester):
        """The guard forbids one path, not SQLite."""
        result = _run(pytester, _OPENS_ITS_OWN_DATABASE)

        result.assert_outcomes(passed=1)

    def test_the_guard_cannot_be_swallowed_by_a_broad_except(self, pytester):
        """`except Exception` is everywhere, including on the read path.

        `_default_crash_report` swallows every `Exception` so a bookkeeping
        write cannot break a listing — and it would have swallowed this too,
        turning a loud guard into a silent pass. Raising outside the `Exception`
        hierarchy is what stops a legitimate catch from hiding the violation.
        """
        result = _run(pytester, _SWALLOWS_EVERY_EXCEPTION)

        result.assert_outcomes(failed=1)
