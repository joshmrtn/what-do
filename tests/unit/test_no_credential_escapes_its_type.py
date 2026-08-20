"""A credential is read at one of two doors, and unwrapped only on purpose.

The `Secret` type makes accidental exposure a type error and the log scrubbing
catches what escapes anyway. Both have the same blind spot: **a value never
minted as a `Secret` is a value the registry has never heard of.** An adapter
that reads its own key from the environment bypasses the type layer and the
value layer together, and nothing else in this codebase would notice. That is
the hole this file exists to close, and it closes it at the moment the module is
added rather than the night it prints a key.

Modelled on `test_network_is_the_only_transport.py`, including the discipline
its docstring names: **forbid operations that are always wrong, never
possession.** An adapter with `api_key: Secret` in its signature, storing it and
passing it on, is untouched by both rules here — and that is most of the surface
and all of it is fine. Removing that false-positive class is what keeps a guard
from being switched off.

Two rules:

* **reading the environment** — confined to the two doors. Everything else
  receives its credential as a parameter. Deliberately broader than "a
  credential-shaped variable name": both doors read through a *variable*
  (`os.environ.get(variable)`), so a rule matching literal names would be blind
  to the only two sites that matter and would fire only on `OLLAMA_HOST` and
  `GEMINI_MODEL`, which are not credentials at all.
* **calling `expose_secret()`** — allowlisted per module, each entry carrying
  its reason. The method name is doing real work: an AST guard cannot see a
  receiver's type, so it matches on the name alone, and `expose_secret` had zero
  occurrences anywhere before this design. **Any hit is a true hit**, which is
  what makes the rule exact rather than heuristic.

Both lists are compared for **equality**, not containment, so an entry that
stops being needed fails here too. A stale exemption is a hole nobody opened on
purpose.

**Honest limit: it catches the accidental, not the adversarial.**
`getattr(s, "expose_" + "secret")()` walks past it, as does a credential
arriving from a config file rather than the environment, or an adapter handed
the environment as a parameter. It is a guard against forgetting, which is the
failure that actually happens.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[2] / "src"

#: May read the environment, and why. These are the two doors: every credential
#: in the process is minted at one of them, which is what makes registration a
#: consequence of arriving rather than a step to remember.
#:
#: Both are *composition*, not behaviour — the whole point is that nothing
#: downstream needs the environment, because it is handed what it needs.
_ENV_READERS = {
    # Reads GEMINI_API_KEY, plus OLLAMA_HOST and GEMINI_MODEL, which are not
    # credentials. `_credential_from_env` is the minting site.
    "config.py",
    # Reads APIFY_API_KEY, AMC_API_KEY and TMDB_READ_ACCESS_TOKEN through
    # `_credential`, which also records the skip when one is absent.
    "composition/batch.py",
}

#: May call `expose_secret()`, and why. Every entry is a credential going onto
#: the wire, and every one of them is a **header** — which is the shape this
#: work moved them all into, and the reason the list is short enough to read.
#:
#: `utils/secret.py` is deliberately absent: it *defines* `expose_secret` and
#: never calls it, and this rule watches calls.
_EXPOSERS = {
    # Apify's token as `Authorization: Bearer`. Their own docs recommend the
    # header over the query parameter.
    "ingestion/social/apify.py",
    # AMC's vendor key as `X-AMC-Vendor-Key`. It has always been a header.
    "ingestion/movies/amc.py",
    # TMDb's v4 read access token as `Authorization: Bearer`, for both the
    # search and the detail request.
    "enrichment/movies.py",
    # Handed to `genai.Client`, which owns the transport from there. The SDK
    # takes a plain string and there is nowhere else to put it.
    "utils/gemini_client.py",
}


def _reads_environment(tree: ast.AST) -> bool:
    """Whether this module reaches for the process environment.

    Catches `os.environ` in every form it is spelled — attribute, `.get(...)`,
    subscript — because they all contain the same `os.environ` attribute node,
    and `os.getenv(...)` alongside it. `from os import environ` is caught at the
    import, rather than by matching a bare `environ` name: `batch.py` binds a
    local called exactly that, and a rule that fired on the name would be
    reporting a parameter it was handed.
    """
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr in {"environ", "getenv"}:
            if isinstance(node.value, ast.Name) and node.value.id == "os":
                return True
        if isinstance(node, ast.ImportFrom) and node.module == "os":
            if any(alias.name in {"environ", "getenv"} for alias in node.names):
                return True
    return False


def _exposes_a_secret(tree: ast.AST) -> bool:
    """Whether this module calls `expose_secret()` on anything."""
    return any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "expose_secret"
        for node in ast.walk(tree)
    )


# ---------------------------------------------------------------------------
# The rules are pure functions, so they are tested before being trusted
# ---------------------------------------------------------------------------
#
# A guard that has only ever been run over a tree it passes has never been shown
# to fire. These snippets are what say it can — and they double as documentation
# of what a correct module looks like.


@pytest.mark.parametrize(
    "source",
    [
        "import os\nk = os.environ['NEWTHING_API_KEY']",
        "import os\nk = os.environ.get('NEWTHING_API_KEY')",
        "import os\nk = os.getenv('NEWTHING_API_KEY')",
        # The form both doors actually use, and the one a name rule cannot see.
        "import os\ndef read(name):\n    return os.environ.get(name)",
        "from os import environ\nk = environ.get('NEWTHING_API_KEY')",
        "from os import getenv\nk = getenv('NEWTHING_API_KEY')",
    ],
)
def test_reaching_for_the_environment_is_seen(source):
    assert _reads_environment(ast.parse(source))


@pytest.mark.parametrize(
    "source",
    [
        # Possession, which is never the violation: this is what every correct
        # credential-handling module looks like.
        "class A:\n    def __init__(self, api_key: Secret):\n        self._key = api_key",
        # A local that happens to be called `environ`, handed in as a parameter.
        "def build(environ):\n    return environ.get('APIFY_API_KEY')",
        # Naming the type, and a variable whose name looks like a credential.
        "def f(token: Secret) -> Secret:\n    return token",
        "import os\nos.path.join('a', 'b')",
    ],
)
def test_an_innocent_module_is_left_alone(source):
    assert not _reads_environment(ast.parse(source))


@pytest.mark.parametrize(
    "source",
    [
        "h = {'Authorization': f'Bearer {key.expose_secret()}'}",
        "c = genai.Client(api_key=self._token.expose_secret())",
        "x = a.b.c.expose_secret()",
    ],
)
def test_unwrapping_a_secret_is_seen(source):
    assert _exposes_a_secret(ast.parse(source))


@pytest.mark.parametrize(
    "source",
    [
        # Defining the method is not calling it — which is why `secret.py`
        # needs no entry on the list.
        "class Secret:\n    def expose_secret(self):\n        return self._value",
        # Passing the credential on, untouched. The common correct case.
        "def build(key: Secret):\n    return Adapter(key)",
        # A docstring mentioning it, which is how this project documents it.
        "def f():\n    'a caller sending one says expose_secret()'",
    ],
)
def test_holding_a_secret_is_not_unwrapping_it(source):
    assert not _exposes_a_secret(ast.parse(source))


# ---------------------------------------------------------------------------
# The rules, run over the tree
# ---------------------------------------------------------------------------


def _survey() -> tuple[set[str], set[str]]:
    """Every module in `src/` that reads the environment, or exposes a secret."""
    readers: set[str] = set()
    exposers: set[str] = set()

    for path in sorted(SRC.rglob("*.py")):
        relative = str(path.relative_to(SRC))
        tree = ast.parse(path.read_text())
        if _reads_environment(tree):
            readers.add(relative)
        if _exposes_a_secret(tree):
            exposers.add(relative)

    return readers, exposers


def test_only_the_two_doors_read_the_environment():
    """A new adapter reading its own key fails here.

    That is the hole the other two layers cannot see. A value read straight from
    the environment was never minted as a `Secret`, so it has no type protecting
    it and the registry has never heard of it — the log scrubbing that would
    catch any *other* provider's credential is silent for this one.

    If the new reader is legitimate it does not go on the list: it takes its
    credential as a parameter, which is what the list exists to make unavoidable.
    """
    readers, _ = _survey()

    assert readers == _ENV_READERS, (
        f"reading the environment outside the two doors: "
        f"{sorted(readers - _ENV_READERS)}; "
        f"listed but no longer reading: {sorted(_ENV_READERS - readers)}"
    )


def test_only_a_declared_site_unwraps_a_credential():
    """Exposure is meant to be a deliberate act, and this is what makes it one.

    Adding an entry is cheap, and that is the design: the cost is having to
    write down why, at the moment somebody is thinking about it, rather than
    discovering later that a credential went somewhere nobody chose.
    """
    _, exposers = _survey()

    assert exposers == _EXPOSERS, (
        f"unwrapping a credential outside a declared site: "
        f"{sorted(exposers - _EXPOSERS)}; "
        f"listed but no longer unwrapping: {sorted(_EXPOSERS - exposers)}"
    )
