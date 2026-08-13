"""Unit tests for the live-versus-repo schema comparison.

The apparent paradox — how do you test a checker whose purpose is to inspect a
file no test may open — dissolves once the comparison is separated from the
thing compared. `compare_schema` takes two connections and knows nothing about
which is "live", so every finding is reachable from two databases built in
`tmp_path`. Nothing is faked; both sides are real SQLite. What remains untested
is the one line that passes the real path in.
"""

from __future__ import annotations

import sqlite3

import pytest

from src.storage.schema_check import check_database, check_references, compare_schema
from src.storage.sqlite.connection import init_db


@pytest.fixture
def expected(tmp_path) -> sqlite3.Connection:
    return sqlite3.connect(tmp_path / "expected.db")


@pytest.fixture
def actual(tmp_path) -> sqlite3.Connection:
    return sqlite3.connect(tmp_path / "actual.db")


def _events(conn: sqlite3.Connection, body: str) -> None:
    conn.execute(f"CREATE TABLE events ({body})")


_BASE = "id TEXT PRIMARY KEY, title TEXT, created_at TEXT NOT NULL"


def test_identical_schemas_produce_nothing(expected, actual):
    """The contract the whole design rests on: any output at all is a problem,
    so silence has to mean silence."""
    _events(expected, _BASE)
    _events(actual, _BASE)

    assert compare_schema(expected, actual) == []


def test_a_column_missing_from_actual_is_reported(expected, actual):
    """The case that kills a batch: `_SCHEMA` grew a column, the live database
    never got the ALTER, and every INSERT naming it fails."""
    _events(expected, _BASE + ", extraction_degradation TEXT")
    _events(actual, _BASE)

    findings = compare_schema(expected, actual)

    assert len(findings) == 1
    assert "extraction_degradation" in findings[0].message
    assert "events" in findings[0].message


def test_a_column_only_in_actual_is_reported(expected, actual):
    """Someone changed one artefact and not the other. Which one is right is
    unknown from here, which is exactly why it wants a human."""
    _events(expected, _BASE)
    _events(actual, _BASE + ", leftover TEXT")

    findings = compare_schema(expected, actual)

    assert len(findings) == 1
    assert "leftover" in findings[0].message


def test_a_type_difference_is_reported(expected, actual):
    _events(expected, "id TEXT PRIMARY KEY, weight REAL")
    _events(actual, "id TEXT PRIMARY KEY, weight TEXT")

    findings = compare_schema(expected, actual)

    assert len(findings) == 1
    assert "weight" in findings[0].message


def test_a_notnull_difference_is_reported(expected, actual):
    """A NOT NULL that exists only in `_SCHEMA` means the live database quietly
    accepts rows every test rejects."""
    _events(expected, "id TEXT PRIMARY KEY, base_score REAL NOT NULL")
    _events(actual, "id TEXT PRIMARY KEY, base_score REAL")

    findings = compare_schema(expected, actual)

    assert len(findings) == 1
    assert "base_score" in findings[0].message


def test_a_default_difference_is_reported(expected, actual):
    _events(expected, "id TEXT PRIMARY KEY, tag_confidence REAL DEFAULT 1.0")
    _events(actual, "id TEXT PRIMARY KEY, tag_confidence REAL DEFAULT 0.0")

    findings = compare_schema(expected, actual)

    assert len(findings) == 1
    assert "tag_confidence" in findings[0].message


def test_a_column_order_difference_is_reported(expected, actual):
    """`event_scores` drifted this way for a month. Benign only while no
    `SELECT *` exists in `src/`, which is not a property anyone remembers."""
    _events(expected, "id TEXT PRIMARY KEY, match TEXT, tag_confidence REAL")
    _events(actual, "id TEXT PRIMARY KEY, tag_confidence REAL, match TEXT")

    findings = compare_schema(expected, actual)

    assert len(findings) == 1
    assert "order" in findings[0].message.lower()


def test_a_table_missing_from_actual_is_reported(expected, actual):
    _events(expected, _BASE)

    findings = compare_schema(expected, actual)

    assert len(findings) == 1
    assert "events" in findings[0].message


def test_a_table_only_in_actual_is_reported(expected, actual):
    """The repo stopped creating something without dropping it."""
    _events(actual, _BASE)

    findings = compare_schema(expected, actual)

    assert len(findings) == 1
    assert "events" in findings[0].message


def test_sqlite_internal_tables_are_ignored(expected, actual):
    """`sqlite_sequence` is created by SQLite the moment an AUTOINCREMENT table
    is, and is not ours to reconcile.

    Asymmetric on purpose. The first cut of this test declared AUTOINCREMENT on
    *both* sides, so both had `sqlite_sequence` and the assertion held whether
    or not anything was filtered — it survived deleting the filter entirely.
    Here only `expected` has it, while `t` itself compares identical, because
    `table_info` reports `INTEGER PRIMARY KEY` for both forms.
    """
    expected.execute("CREATE TABLE t (id INTEGER PRIMARY KEY AUTOINCREMENT)")
    actual.execute("CREATE TABLE t (id INTEGER PRIMARY KEY)")

    assert "sqlite_sequence" in [
        row[0] for row in expected.execute("SELECT name FROM sqlite_master")
    ], "the fixture stopped producing the internal table this test is about"
    assert compare_schema(expected, actual) == []


def test_every_difference_is_reported_not_just_the_first(expected, actual):
    """A checker that stops at the first finding turns one migration into
    several rounds of run-fix-rerun."""
    _events(expected, _BASE + ", a TEXT, b TEXT")
    _events(actual, _BASE)

    assert len(compare_schema(expected, actual)) == 2


def test_findings_name_their_table(expected, actual):
    """Nineteen tables share column names — `event_id` is on six of them."""
    expected.execute("CREATE TABLE event_scores (event_id TEXT, extra TEXT)")
    actual.execute("CREATE TABLE event_scores (event_id TEXT)")

    assert compare_schema(expected, actual)[0].table == "event_scores"


class TestReferentialIntegrity:
    """A database can match `_SCHEMA` column for column and have lost its
    references entirely.

    Measured: rebuilding `event_scores` via `RENAME TO` repointed
    `score_reasons`'s foreign key at the renamed table, which was then dropped —
    7,297 dangling references, and every column on every table still identical
    to a fresh build. The comparison above reported clean.
    """

    def test_a_dangling_reference_is_reported(self, actual):
        actual.execute("CREATE TABLE parent (id TEXT PRIMARY KEY)")
        actual.execute(
            "CREATE TABLE child (id TEXT PRIMARY KEY, "
            "parent_id TEXT REFERENCES parent(id))"
        )
        actual.execute("PRAGMA foreign_keys = OFF")
        actual.execute("INSERT INTO child VALUES ('c1', 'nobody')")

        findings = check_references(actual)

        assert len(findings) == 1
        assert "child" in findings[0].message

    def test_intact_references_produce_nothing(self, actual):
        actual.execute("CREATE TABLE parent (id TEXT PRIMARY KEY)")
        actual.execute(
            "CREATE TABLE child (id TEXT PRIMARY KEY, "
            "parent_id TEXT REFERENCES parent(id))"
        )
        actual.execute("INSERT INTO parent VALUES ('p1')")
        actual.execute("INSERT INTO child VALUES ('c1', 'p1')")

        assert check_references(actual) == []

    def test_a_reference_to_a_dropped_table_is_reported(self):
        """The exact shape of the `event_scores` breakage: the FK names a table
        that no longer exists at all."""
        conn = sqlite3.connect(":memory:")
        conn.execute("CREATE TABLE parent (id TEXT PRIMARY KEY)")
        conn.execute(
            "CREATE TABLE child (id TEXT PRIMARY KEY, "
            "parent_id TEXT REFERENCES parent(id))"
        )
        conn.execute("PRAGMA foreign_keys = OFF")
        conn.execute("INSERT INTO child VALUES ('c1', 'p1')")
        conn.execute("DROP TABLE parent")

        findings = check_references(conn)

        assert findings
        assert "child" in findings[0].message

    def test_the_count_is_reported_not_every_row(self, actual):
        """7,297 findings is not a report, it is a wall. One line per table
        pairing, with the count."""
        actual.execute("CREATE TABLE parent (id TEXT PRIMARY KEY)")
        actual.execute(
            "CREATE TABLE child (id TEXT PRIMARY KEY, "
            "parent_id TEXT REFERENCES parent(id))"
        )
        actual.execute("PRAGMA foreign_keys = OFF")
        for i in range(50):
            actual.execute("INSERT INTO child VALUES (?, 'nobody')", (f"c{i}",))

        findings = check_references(actual)

        assert len(findings) == 1
        assert "50" in findings[0].message


class TestCheckingADatabaseOnDisk:
    """The thin edge: build a fresh database, compare the real one to it.

    This is the only part a test cannot reach past — it names a real path — so
    it is kept to almost nothing, and what it delegates to is tested above.
    """

    def test_a_freshly_initialised_database_is_clean(self, tmp_path):
        db = tmp_path / "fresh.db"
        init_db(db)

        assert check_database(db) == []

    def test_a_column_difference_is_found(self, tmp_path):
        """That `check_database` reaches the comparison at all; which direction
        of difference it reports is `compare_schema`'s business and is covered
        above, both ways.

        Adds rather than drops because `ALTER TABLE ... DROP COLUMN` fails on
        our tables: SQLite rewrites the stored SQL and chokes on the `--`
        comments `_SCHEMA` keeps inside its CREATE statements.
        """
        db = tmp_path / "drifted.db"
        init_db(db)
        conn = sqlite3.connect(db)
        conn.execute("ALTER TABLE events ADD COLUMN not_in_the_repository TEXT")
        conn.commit()
        conn.close()

        findings = check_database(db)

        assert any("not_in_the_repository" in f.message for f in findings)

    def test_a_dangling_reference_is_found(self, tmp_path):
        """A column comparison cannot see this, which is why both run."""
        db = tmp_path / "dangling.db"
        init_db(db)
        conn = sqlite3.connect(db)
        conn.execute("PRAGMA foreign_keys = OFF")
        conn.execute(
            "INSERT INTO event_tags (event_id, position, tag, weight) "
            "VALUES ('no-such-event', 0, 'jazz', 1.0)"
        )
        conn.commit()
        conn.close()

        findings = check_database(db)

        assert any("event_tags" in f.message for f in findings)
