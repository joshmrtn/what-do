"""Compare a database's schema against the one `_SCHEMA` describes.

`_SCHEMA` is all `CREATE TABLE IF NOT EXISTS`, so a new **column** serves every
freshly initialised database — every test — while the live file is untouched
until a hand migration runs. The suite is green either way and no test can see
the difference. This is the only check that can.

The comparison takes two connections and knows nothing about which is "live", so
every finding it can produce is reachable from two databases a test builds in a
temporary directory. Only the caller knows which side is real.

There is deliberately **no allowlist**. The one known-benign difference —
`event_scores`'s column order, left by a hand migration — was fixed rather than
described, which leaves this with the strongest contract available: any output
at all is a problem.
"""

from __future__ import annotations

import sqlite3
import tempfile
from dataclasses import dataclass
from pathlib import Path

#: One column as `PRAGMA table_info` gives it, minus the positional index —
#: which the list order carries instead. Named types rather than `object`
#: because the name is sorted on, and `object` is not orderable.
Column = tuple[str, str, int, str | None, int]

#: SQLite's own bookkeeping. `sqlite_sequence` appears the moment an
#: AUTOINCREMENT table receives a row, so it can exist on one side and not the
#: other for reasons that are nothing to do with us.
_INTERNAL_PREFIX = "sqlite_"


@dataclass(frozen=True)
class Finding:
    """One difference between the two schemas.

    `table` is separate from the message because nineteen tables share column
    names — `event_id` is on six of them — and a finding that does not say which
    one it means costs a grep to act on.
    """

    table: str
    message: str


def compare_schema(
    expected: sqlite3.Connection, actual: sqlite3.Connection
) -> list[Finding]:
    """Every way `actual`'s schema differs from `expected`'s.

    Args:
        expected: A database built by `init_db` — what the repository says.
        actual: The database to check.

    Returns:
        One finding per difference, in table then column order. Empty means the
        two agree column for column, which is the only acceptable result.
    """
    expected_tables = _tables(expected)
    actual_tables = _tables(actual)
    findings: list[Finding] = []

    for table in sorted(set(expected_tables) - set(actual_tables)):
        findings.append(Finding(table, f"table {table!r} is missing"))
    for table in sorted(set(actual_tables) - set(expected_tables)):
        findings.append(
            Finding(table, f"table {table!r} exists but the repository does not create it")
        )

    for table in sorted(set(expected_tables) & set(actual_tables)):
        findings.extend(_compare_table(table, expected_tables[table], actual_tables[table]))

    return findings


def _compare_table(
    table: str, expected: list[Column], actual: list[Column]
) -> list[Finding]:
    """Differences within one table the two sides share."""
    expected_by_name = {row[0]: row for row in expected}
    actual_by_name = {row[0]: row for row in actual}
    findings: list[Finding] = []

    for name in [c[0] for c in expected if c[0] not in actual_by_name]:
        findings.append(
            Finding(table, f"{table}.{name} is in the repository but missing here")
        )
    for name in [c[0] for c in actual if c[0] not in expected_by_name]:
        findings.append(
            Finding(table, f"{table}.{name} exists but the repository does not declare it")
        )

    for name, expected_row in expected_by_name.items():
        actual_row = actual_by_name.get(name)
        if actual_row is not None and actual_row[1:] != expected_row[1:]:
            findings.append(
                Finding(
                    table,
                    f"{table}.{name} differs: repository {_describe(expected_row)}, "
                    f"here {_describe(actual_row)}",
                )
            )

    # Only worth saying when the columns themselves agree; otherwise the
    # missing-column findings above already explain the mismatch, and adding
    # "different order" to them is noise rather than a second problem.
    expected_names = [c[0] for c in expected]
    actual_names = [c[0] for c in actual]
    if expected_names != actual_names and sorted(expected_names) == sorted(actual_names):
        findings.append(
            Finding(
                table,
                f"{table}: same columns in a different order — "
                f"repository {expected_names}, here {actual_names}",
            )
        )

    return findings


def _describe(row: Column) -> str:
    """A column's type, nullability and default, as a person would read them."""
    _, column_type, notnull, default, _ = row
    parts = [column_type or "(no type)"]
    if notnull:
        parts.append("NOT NULL")
    if default is not None:
        parts.append(f"DEFAULT {default}")
    return " ".join(parts)


def _tables(conn: sqlite3.Connection) -> dict[str, list[Column]]:
    """Every table's columns, in declared order, keyed by table name.

    Each column is `(name, type, notnull, default, pk)` — `PRAGMA table_info`
    without its positional index, which is carried by the list order instead.
    """
    names = [
        row[0]
        for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        if not str(row[0]).startswith(_INTERNAL_PREFIX)
    ]
    return {
        name: [
            (
                str(row[1]),
                str(row[2]),
                int(row[3]),
                None if row[4] is None else str(row[4]),
                int(row[5]),
            )
            for row in conn.execute(f"PRAGMA table_info({name})")
        ]
        for name in names
    }


def check_references(conn: sqlite3.Connection) -> list[Finding]:
    """Every foreign key in `conn` that points at a row which is not there.

    Separate from the column comparison because it is a different property, and
    the reason it exists at all is that the two were once conflated: rebuilding
    `event_scores` through `ALTER TABLE ... RENAME TO` repointed
    `score_reasons`'s foreign key at the renamed table, which was then dropped.
    Every column on every table still matched a freshly initialised database
    exactly. `compare_schema` reported clean over 7,297 dangling references.

    Reported one line per table pairing with a count, not one per row: seven
    thousand findings is a wall, not a report.
    """
    counts: dict[tuple[str, str], int] = {}
    for child, _rowid, parent, _fkid in conn.execute("PRAGMA foreign_key_check"):
        counts[(str(child), str(parent))] = counts.get((str(child), str(parent)), 0) + 1

    return [
        Finding(
            child,
            f"{child}: {count} row(s) reference {parent!r}, which does not have them",
        )
        for (child, parent), count in sorted(counts.items())
    ]


def check_database(db_path: Path | str) -> list[Finding]:
    """Everything wrong with the schema at `db_path`, or nothing.

    Builds a throwaway database from `_SCHEMA` and compares against it, then
    checks referential integrity. The two are different properties and a
    database can pass either while failing the other.

    The import is local: `init_db` lives in the module this one is checking, and
    at module scope the pair would import each other.

    Args:
        db_path: The database to check. Opened read-only — a checker that could
            write is a checker that could "fix" something unasked.

    Returns:
        One finding per problem. Empty is the only acceptable result; there is
        no allowlist, so anything here is worth a person's attention.
    """
    from src.storage.sqlite.connection import init_db

    actual = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        with tempfile.TemporaryDirectory() as tmp:
            fresh_path = Path(tmp) / "fresh.db"
            init_db(fresh_path)
            expected = sqlite3.connect(fresh_path)
            try:
                findings = compare_schema(expected, actual)
            finally:
                expected.close()
        return findings + check_references(actual)
    finally:
        actual.close()


def format_findings(db_path: Path | str, findings: list[Finding]) -> str:
    """The findings as a person reads them, with what to do about them."""
    if not findings:
        return f"{db_path}: schema matches the repository"
    lines = [f"{db_path}: {len(findings)} schema problem(s):"]
    lines.extend(f"  {finding.message}" for finding in findings)
    lines.append(
        "\nThe repository and this database disagree. Neither is automatically "
        "right — a migration may be missing, or `_SCHEMA` may have been changed "
        "without one. See ~/claude-docs/what-do/migrations/."
    )
    return "\n".join(lines)


def run() -> int:
    """Entry point for `what-do-check-schema`."""
    import argparse

    from src.storage.sqlite.connection import DEFAULT_DB_PATH

    parser = argparse.ArgumentParser(
        prog="what-do-check-schema",
        description="Compare a database's schema against the one the repository declares.",
    )
    parser.add_argument("db", nargs="?", default=str(DEFAULT_DB_PATH))
    args = parser.parse_args()

    findings = check_database(args.db)
    print(format_findings(args.db, findings))
    return 1 if findings else 0
