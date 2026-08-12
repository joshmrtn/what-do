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

    `CLAUDE.md`: "The CLI reads only precomputed data from SQLite. No LLM or
    network calls during interactive use." Importing is not calling, but it is
    the wrong direction, and it is measurable: a convenience re-export in
    `composition/__init__.py` briefly made importing `composition.storage`
    execute `batch`, taking the CLI from 41 loaded modules to 105 — every
    adapter and the Ollama client — while every test stayed green.

    Asserted on the loaded module graph rather than on imports, because the
    regression came in through a package `__init__`, which no import statement
    in `cli.py` would have shown.
    """
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, src.presentation.cli;"
            "print(any('ollama' in m for m in sys.modules));"
            "print('requests' in sys.modules)",
        ],
        capture_output=True,
        text=True,
        cwd=SRC.parent,
    )

    assert result.returncode == 0, result.stderr
    loads_ollama, loads_requests = result.stdout.split()
    assert loads_ollama == "False", "the CLI imports the Ollama client"
    assert loads_requests == "False", "the CLI imports the HTTP layer"
