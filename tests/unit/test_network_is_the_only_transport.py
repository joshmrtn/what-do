"""Nothing performs a request except through the politeness adapter.

Eleven modules made HTTP calls and exactly one of them was polite. The fix was
not eleven fixes: doing it caller by caller is how eleven slightly different
politeness policies appear with no way to tell which module is the impolite one.
It was one adapter every caller goes through — and this is what keeps it that
way, structurally, the way `composition/storage.py` made repositories
un-bypassable. A rule that has to be remembered at every new call site is a rule
that will be forgotten, and this one already was.

**It forbids a call, not an import.** Ten modules in `src/` import a transport
and most of them are behaving correctly: naming `requests.Session` as a
parameter type and catching `requests.RequestException` is exactly what a caller
*should* do once it has been moved onto the adapter. A guard on the import would
fire on seven modules that are right, which is how a guard gets switched off.

Three rules, because there are three ways past the adapter:

* **performing a request** — `requests.get(...)`, `urlopen(...)`. Only the
  adapter, and localhost;
* **building a transport client** — a `Session`, an SDK `Client`. Whoever holds
  one can use it, so only a composition root may make one;
* **holding a client type without reaching for the policy** — the gap the first
  two leave, because an injected session's `.get` is indistinguishable from a
  dict's. A module that names one must import `src.network`.

The lists below are the record of why each exception exists, which is the part
that stops them growing quietly. Every entry carries its reason.
"""

from __future__ import annotations

import ast
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "src"

#: Top-level names that are somebody else's transport. `genai` is here because
#: watching only `requests` would leave the next SDK-based provider a way in,
#: which is how `gemini_client.py` was missed by the original survey.
_TRANSPORTS = {"requests", "urllib", "urllib3", "httpx", "http", "genai", "aiohttp"}

#: Attributes that put a request on the wire.
_VERBS = {
    "get",
    "post",
    "put",
    "patch",
    "delete",
    "head",
    "options",
    "request",
    "send",
    "urlopen",
    "urlretrieve",
}

#: Attributes that hand back something able to perform requests later. Calling
#: one is not itself a request, which is why it is judged separately — but
#: whoever holds the result can reach the network without saying so.
_CLIENTS = {
    "Session",
    "Client",
    "AsyncClient",
    "ClientSession",
    "PoolManager",
    "HTTPConnection",
    "HTTPSConnection",
}

#: May perform a request directly, and why.
#:
#: **Empty since 2026-08-20**, and that is the point: `utils/ollama_client.py`
#: was the last entry, exempt as *localhost*. Locality excuses spacing — there is
#: no third party at this address — and excuses nothing about a dropped request,
#: so the model client now performs through an injected session behind the policy
#: like everything else (#36). An exemption asserted by module path could not see
#: OLLAMA_HOST being repointed at another machine; an address in `network.hosts`
#: can.
#:
#: `network/http.py` is deliberately **absent** for a different reason: the
#: adapter performs through the session it was handed, so it never calls the
#: module and this rule never sees it. That is the same blindness rule three
#: exists to cover, and it is why the adapter is not privileged here — if it ever
#: reaches for `requests` directly, it earns an entry like anyone else.
_PERFORMERS: set[str] = set()

#: May construct a transport client, and why.
_TRANSPORT_BUILDERS = {
    # The composition roots. Building the objects they inject is their job, and
    # it is the reason the eight adapters no longer default to a bare session.
    "composition/network.py",
    "composition/batch.py",
    # The read path, which builds the model client the CLI embeds through.
    "composition/view.py",
    # The bench's own root. It talks only to the model under test, and its
    # numbers are deliberately not the pipeline's — one attempt, and an hour to
    # answer, because a retry would fold two runs into one measurement.
    "bench/cli.py",
    # The SDK boundary itself: it builds the vendor client lazily and every call
    # it makes on it goes through the policy. Same standing as `network/http.py`
    # has for `requests` — the module that owns a transport may name it.
    "utils/gemini_client.py",
}


def _transport_names(tree: ast.AST) -> tuple[set[str], set[str], bool]:
    """What this module bound to a transport, and whether it knows the policy.

    Returns:
        `(aliases, direct, reaches_for_the_policy)` — module aliases such as the
        `requests` in `import requests`, names imported straight out of one such
        as the `urlopen` in `from urllib.request import urlopen`, and whether the
        module imports anything from `src.network`.
    """
    aliases: set[str] = set()
    direct: set[str] = set()
    policy = False

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root in _TRANSPORTS:
                    aliases.add(alias.asname or root)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module.startswith("src.network"):
                policy = True
                continue
            root = module.split(".")[0]
            if root == "google":
                for alias in node.names:
                    if alias.name in _TRANSPORTS:
                        aliases.add(alias.asname or alias.name)
            elif root in _TRANSPORTS:
                for alias in node.names:
                    if alias.name in _VERBS or alias.name in _CLIENTS:
                        direct.add(alias.asname or alias.name)
                    else:
                        aliases.add(alias.asname or alias.name)

    return aliases, direct, policy


def _root_of(node: ast.expr) -> str | None:
    """The name an attribute chain starts from, for `a.b.c` → `a`."""
    while isinstance(node, ast.Attribute):
        node = node.value
    return node.id if isinstance(node, ast.Name) else None


def _survey() -> tuple[dict[str, set[str]], dict[str, set[str]], set[str]]:
    """Every module in `src/`, by what it does with a transport.

    Returns:
        `(performers, builders, holders)` — modules calling a request verb,
        modules constructing a client, and modules naming a client type without
        importing `src.network`. Keys are `src`-relative paths; the values name
        what was found, so a failure says which call it is rather than only where.
    """
    performers: dict[str, set[str]] = {}
    builders: dict[str, set[str]] = {}
    holders: set[str] = set()

    for path in sorted(SRC.rglob("*.py")):
        relative = path.relative_to(SRC).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8"))
        aliases, direct, policy = _transport_names(tree)
        if not aliases and not direct:
            continue

        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute):
                if _root_of(node.value) in aliases or (
                    isinstance(node.value, ast.Name) and node.value.id in aliases
                ):
                    if node.attr in _CLIENTS and not policy:
                        holders.add(relative)

            if not isinstance(node, ast.Call):
                continue

            called = node.func
            if isinstance(called, ast.Attribute) and _root_of(called) in aliases:
                name, attribute = f"{_root_of(called)}.{called.attr}", called.attr
            elif isinstance(called, ast.Name) and called.id in direct:
                name, attribute = called.id, called.id
            else:
                continue

            if attribute in _VERBS:
                performers.setdefault(relative, set()).add(name)
            elif attribute in _CLIENTS:
                builders.setdefault(relative, set()).add(name)

    return performers, builders, holders


def test_only_the_adapter_and_localhost_perform_a_request():
    """A new `requests.get` anywhere in `src/` fails here.

    If the new caller is legitimate it does not go on the list — it goes through
    `RequestPolicy`, which is what the list exists to make unavoidable. The two
    entries are the adapter and a caller that never leaves this machine.
    """
    performers, _, _ = _survey()

    assert set(performers) == _PERFORMERS, (
        f"performing requests outside the adapter: "
        f"{ {k: sorted(v) for k, v in performers.items() if k not in _PERFORMERS} }"
    )


def test_only_a_composition_root_builds_a_transport_client():
    """Whoever builds a session can use it without anyone seeing.

    This is the rule that survives a transport we have not met: an SDK client is
    constructed the same way a `Session` is, and the module that ends up holding
    one is the module that can quietly go round the policy.
    """
    _, builders, _ = _survey()

    assert set(builders) == _TRANSPORT_BUILDERS, (
        f"building a transport client outside a root: "
        f"{ {k: sorted(v) for k, v in builders.items() if k not in _TRANSPORT_BUILDERS} }"
    )


def test_a_module_naming_a_client_type_reaches_for_the_policy():
    """The gap the other two rules leave.

    An injected session's `.get(...)` is indistinguishable from a dict's, so no
    guard can catch it directly. What *is* visible is that the module names the
    type at all — and a module holding a session with no idea the policy exists
    is the shape every impolite caller had before phase 2.
    """
    _, _, holders = _survey()

    assert holders <= _TRANSPORT_BUILDERS, (
        f"holding a transport client without importing src.network: "
        f"{sorted(holders - _TRANSPORT_BUILDERS)}"
    )
