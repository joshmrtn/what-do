"""Only a composition root may name a concrete storage implementation.

The repository split exists so that what a caller gets is decided in one place.
That guarantee used to be enforced by nothing: every repository accepted `None`
and built its own SQLite implementation, so a module that forgot to inject got a
database anyway and looked correct doing it.

Removing those fallbacks (storage F2) made composition the single point of
failure. This keeps it that way — a rule that has to be remembered at every new
import is a rule that will be forgotten.
"""

from __future__ import annotations

import ast
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "src"

#: Modules permitted to name a concrete implementation, and why.
_ROOTS = {
    # The batch composition root.
    "composition.py",
    # The view composition root: it reads `--db`, defaults it and checks
    # `has_schema` before reading, so naming the database is its job. Folding it
    # into the batch root would make the query path import the LLM client it is
    # forbidden to use — measured at 56 extra modules. See F5 in the roadmap.
    "presentation/cli.py",
    # A deliberate escape hatch, documented where it lives: candidate writes are
    # interleaved with the ingestion accept loop and batched into one
    # transaction, so they take the caller's connection rather than opening a
    # second one that would wait on a lock this same call stack holds.
    "ingestion/ingestion_service.py",
}


def _modules_importing_sqlite() -> set[str]:
    """Every `src` module that imports from `storage.sqlite`, by relative path."""
    offenders: set[str] = set()
    for path in SRC.rglob("*.py"):
        relative = path.relative_to(SRC).as_posix()
        if relative.startswith("storage/sqlite/"):
            continue  # the implementation is allowed to know itself
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.ImportFrom) and (node.module or "").startswith(
                "src.storage.sqlite"
            ):
                offenders.add(relative)
            elif isinstance(node, ast.Import):
                if any(a.name.startswith("src.storage.sqlite") for a in node.names):
                    offenders.add(relative)
    return offenders


def test_only_the_roots_name_a_concrete_implementation():
    """A new import of `storage.sqlite` outside a root fails here.

    If the new one is legitimate, add it to `_ROOTS` *with its reason* — the
    list is the record of why each exception exists, which is the part that
    stops it growing quietly.
    """
    assert _modules_importing_sqlite() == _ROOTS


def test_the_scheduler_names_no_concrete_implementation():
    """The debt recorded against storage phase B, stated as a test.

    "Phase F is not complete until `src/scheduler.py` imports no concrete
    repository and `composition.py` constructs it."
    """
    assert "scheduler.py" not in _modules_importing_sqlite()
