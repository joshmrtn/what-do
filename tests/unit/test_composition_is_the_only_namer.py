"""Only a composition root may name a concrete storage implementation.

The repository split exists so that what a caller gets is decided in one place.
That guarantee used to be enforced by nothing: every repository accepted `None`
and built its own SQLite implementation, so a module that forgot to inject got a
database anyway and looked correct doing it.

Removing those fallbacks (storage F2) made composition the single point of
failure. This keeps it that way — a rule that has to be remembered at every new
import is a rule that will be forgotten.

Two different things live under `storage/sqlite/` and they carry different
rules, which is why there are two lists below:

* a **repository implementation** decides what a caller gets, so only a
  composition root may name one;
* `connection.py` is **infrastructure** — `connect`, `transaction`, `init_db`,
  `has_schema`. Reaching for it is not the same offence, but it is still debt,
  because a module holding a raw connection is a module the storage layer does
  not fully cover.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "src"

_CONNECTION = "src.storage.sqlite.connection"

#: Modules permitted to name a concrete *implementation*, and why.
_ROOTS = {
    # The one storage factory. Both roots build through it, which is what stops
    # them drifting: the batch used to pass `config.models.embeddings` to the
    # event repository while the CLI took the default.
    "composition/storage.py",
    # A deliberate escape hatch, documented where it lives: candidate writes are
    # interleaved with the ingestion accept loop and batched into one
    # transaction, so they take the caller's connection rather than opening a
    # second one that would wait on a lock this same call stack holds.
    "ingestion/ingestion_service.py",
}

#: Modules still holding a raw SQLite connection. **This list should only ever
#: shrink.** Each entry is debt with a known owner:
_RAW_CONNECTION = {
    # Entry points. They name the database because that is their job —
    # `DEFAULT_DB_PATH`, `init_db`, `has_schema`.
    "scheduler.py",
    "presentation/cli.py",
    # The bench (#2). `DEFAULT_DB_PATH` only: its samples come through
    # `build_view_storage`, the same factory both other roots use, so it names
    # no implementation of its own.
    "bench/cli.py",
    # The schema checker. Not debt: opening a database directly *is* its job —
    # it compares a real file against a throwaway one built from `_SCHEMA`, and
    # a repository would hide the very difference it exists to find. Read-only.
    "storage/schema_check.py",
    # The module-level storage functions the repositories wrap rather than
    # restate. Goes when nothing calls them directly.
    "storage/events.py",
    # The two escape hatches, for the batched-transaction reason above.
    "ingestion/ingestion_service.py",
    "ingestion/venue_discovery.py",
    # Storage E2, deferred by design: this wraps `preference_embeddings_cache`,
    # a table scheduled for deletion by the preference-revision work. Building a
    # repository over it now means building it twice.
    "scoring/preferences.py",
}


def _sqlite_importers() -> tuple[set[str], set[str]]:
    """Modules importing a concrete implementation, and those importing `connect`.

    Returns:
        `(implementations, connections)` as `src`-relative paths. Modules inside
        `storage/sqlite/` are excluded from both — the implementation is allowed
        to know itself.
    """
    implementations: set[str] = set()
    connections: set[str] = set()
    for path in SRC.rglob("*.py"):
        relative = path.relative_to(SRC).as_posix()
        if relative.startswith("storage/sqlite/"):
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            module = getattr(node, "module", None) or ""
            if not isinstance(node, ast.ImportFrom):
                continue
            if module == _CONNECTION:
                connections.add(relative)
            elif module.startswith("src.storage.sqlite"):
                implementations.add(relative)
    return implementations, connections


#: Modules permitted to call `load_events`, the unfiltered reader.
#: **Only the repository that wraps it.**
_EVENT_READERS = {"storage/sqlite/events.py"}

#: The reader that does not filter superseded rows. Narrow on purpose: the
#: module also holds `validate_tag_vectors` (a shared validator) and
#: `load_tag_embeddings` (read by a composition root), and neither can forget a
#: filter it never had.
_UNFILTERED_READER = "load_events"


def _module_level_event_readers() -> set[str]:
    """Modules importing the unfiltered `load_events` directly."""
    found: set[str] = set()
    for path in SRC.rglob("*.py"):
        relative = path.relative_to(SRC).as_posix()
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if not isinstance(node, ast.ImportFrom):
                continue
            if node.module != "src.storage.events":
                continue
            if any(alias.name == _UNFILTERED_READER for alias in node.names):
                found.add(relative)
    return found


def test_only_the_repository_reads_events_through_the_module_functions():
    """`load_events` does not filter superseded rows; `load_all` does.

    Two readers that disagree about which rows exist is one reader too many, and
    the one that forgets is always the one production takes: `--raw` reached
    past the repository to `load_events` as a **default argument**, so every
    test injected its own loader while production ran the unfiltered path (#28).

    The module functions are not going away — they are the row mapping the
    repository wraps rather than restates. What must not come back is a caller
    outside `storage/` reaching for them.
    """
    assert _module_level_event_readers() == _EVENT_READERS


def test_only_the_roots_name_a_concrete_implementation():
    """A new import of a `storage.sqlite` repository outside a root fails here.

    If the new one is legitimate, add it to `_ROOTS` *with its reason* — the
    list is the record of why each exception exists, which is the part that
    stops it growing quietly.
    """
    implementations, _ = _sqlite_importers()

    assert implementations == _ROOTS


def test_the_scheduler_names_no_concrete_implementation():
    """The debt recorded against storage phase B, stated as a test.

    "Phase F is not complete until `src/scheduler.py` imports no concrete
    repository and `composition.py` constructs it."
    """
    implementations, _ = _sqlite_importers()

    assert "scheduler.py" not in implementations


def test_no_new_module_takes_a_raw_connection():
    """The raw-connection list is allowed to shrink and nothing else.

    Separated from the rule above because it is a weaker offence with a
    different fix: these do not decide what a caller gets, they just know the
    storage engine. Each one has an owner recorded beside it.
    """
    _, connections = _sqlite_importers()

    assert connections <= _RAW_CONNECTION, (
        f"new module(s) taking a raw connection: {sorted(connections - _RAW_CONNECTION)}"
    )


def test_the_view_root_does_not_import_the_batch_root():
    """The query path must not meet the LLM client it is forbidden to use.

    `CLAUDE.md`: "The CLI shall make no LLM calls during interactive use."
    Network is expressly allowed — the rule is about a model's *latency*, not
    about sockets. Importing is not calling, but it is
    the wrong direction, and it is measurable: a convenience re-export in
    `composition/__init__.py` briefly made importing `composition.storage`
    execute `batch`, taking the CLI from 41 loaded modules to 105 — every
    adapter and the Ollama client — while every test stayed green.

    Asserted on the loaded module graph rather than on imports, because the
    regression came in through a package `__init__`, which no import statement
    in `cli.py` would have shown.

    **Narrowed when the read-time rescore landed**, and narrowed rather than
    deleted. It used to assert `requests` was absent too, which was accurate
    about a CLI that opened no sockets and was never the architectural rule.
    The rescore refreshes a forecast, so `requests` arrives legitimately;
    measured, that took CLI startup from 0.22s to 0.29s, which is the price of
    the feature and is still snappy. What must never arrive is a *model*: the
    Ollama client, and the extraction provider that would reach for one.
    """
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, src.presentation.cli;"
            "print(any('ollama' in m for m in sys.modules));"
            "print('src.processing.extraction' in sys.modules)",
        ],
        capture_output=True,
        text=True,
        cwd=SRC.parent,
    )

    assert result.returncode == 0, result.stderr
    loads_ollama, loads_extraction = result.stdout.split()
    assert loads_ollama == "False", "the CLI imports the Ollama client"
    assert loads_extraction == "False", "the CLI imports the extraction provider"


#: The one module allowed to rank. Ranking is the pipeline's terminal step, and
#: two roots each calling it would have to agree — forever, with nothing
#: checking — about the scope filter, the argument order and the order the two
#: halves are written in.
_RANKERS = {"composition/pipeline.py"}

#: Where the scope predicate is defined. It used to live in `scheduler.py`, the
#: batch root, which left the view two bad options: import the batch root, or
#: restate the predicate. The second is the "a double may record but it may not
#: reimplement" rule in structural form.
_SCOPE_OWNERS = {"composition/pipeline.py"}


def _modules_calling(method: str) -> set[str]:
    """Modules containing a call to `<anything>.<method>(...)`."""
    found: set[str] = set()
    for path in SRC.rglob("*.py"):
        relative = path.relative_to(SRC).as_posix()
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if isinstance(func, ast.Attribute) and func.attr == method:
                found.add(relative)
    return found


def _modules_defining(*names: str) -> set[str]:
    """Modules defining a top-level function with any of these names."""
    wanted = set(names)
    found: set[str] = set()
    for path in SRC.rglob("*.py"):
        relative = path.relative_to(SRC).as_posix()
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.FunctionDef) and node.name in wanted:
                found.add(relative)
    return found


def test_only_the_pipeline_ranks():
    """A second `.rank(...)` call site fails here rather than drifting quietly.

    The read-time rescore re-runs the tail hours after the batch did. If it
    called the engine itself, the two would share four things by convention —
    the scope predicate, the run date, the argument order, and scores being
    written before rankings because the foreign key refuses the reverse. The
    fourth is the one that fails loudly; the first three fail as a wrong
    ordering nobody can see.
    """
    assert _modules_calling("rank") == _RANKERS


def test_the_scope_predicate_lives_outside_both_roots():
    """`scope_filter` answers "is this worth ranking?" for whoever is ranking.

    Defined in the batch root, it forced the view to import the batch root —
    which `test_the_view_root_does_not_import_the_batch_root` forbids for
    measured reasons — or to write the predicate a second time.
    """
    assert _modules_defining("scope_filter", "scope_floor") == _SCOPE_OWNERS


def test_no_scope_predicate_survives_in_the_scheduler():
    """The underscore-prefixed originals are gone, not shadowed.

    A private copy left behind in `scheduler.py` would satisfy the check above
    while the batch quietly went on using its own.
    """
    assert _modules_defining("_scope_filter", "_scope_floor") == set()
