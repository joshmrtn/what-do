"""A credential that cannot be printed, and a registry of values never to log.

Two layers, and they fail apart.

`Secret` is the **type** layer, and it is the one that matters. A credential
typed `Secret` cannot be interpolated, repr'd, formatted or serialised into
anything — `mypy --strict` names every site that would need `expose_secret()`,
so accidental exposure is a type error and deliberate exposure is one greppable
token. That is what makes a source added next month safe by default rather than
by remembering.

`scrub` is the **value** layer, and it is a backstop. Once `expose_secret()` has
been called the value is an ordinary string, and it can end up inside text we
did not build — `str(requests.HTTPError)` embeds the full URL including the
query string, and a vendor SDK's error may carry anything at all. Matching on
the value means this works for a provider whose parameter names we have never
seen.

**Registering is a side effect of minting**, deliberately. An explicit
`register(...)` call beside every credential is a rule to be remembered at every
new call site, and this module exists because that kind of rule gets forgotten.

Secrecy is a property of the **value**, never of its destination. There is no
localhost exemption here — a local database password is a credential, and
"it's localhost" is in any case a runtime property of a configurable host that
can be repointed with no code change to notice. The politeness adapter's
localhost exemption answers a different question (throttling protects a third
party, and there is no third party) and does not transfer.
"""

from __future__ import annotations

#: What a redacted value renders as. Distinctive enough to grep a log for.
PLACEHOLDER = "…redacted…"

#: Below this length, a value is not searched for in arbitrary text. A
#: three-character credential would redact its own letters out of every line in
#: the batch, which is a denial of service on our own logs. The floor belongs to
#: the backstop only: `Secret` still refuses to render a short value, so the
#: type layer protects every credential whatever its length.
MIN_REDACTABLE_LENGTH = 8


class _Redactions:
    """Values that must never reach a log, in the order they must be replaced.

    Longest first: a credential that contains another as a prefix would
    otherwise have its tail stranded in the log after the inner value was
    replaced.
    """

    def __init__(self) -> None:
        self._values: set[str] = set()
        self._ordered: tuple[str, ...] = ()

    def register(self, value: str) -> None:
        """Record a value to scrub, if it is long enough to search for safely."""
        if len(value) < MIN_REDACTABLE_LENGTH or value in self._values:
            return
        self._values.add(value)
        self._ordered = tuple(sorted(self._values, key=len, reverse=True))

    def scrub(self, text: str) -> str:
        """Replace every registered value in `text` with the placeholder."""
        for value in self._ordered:
            text = text.replace(value, PLACEHOLDER)
        return text

    def snapshot(self) -> frozenset[str]:
        """The registered values, for a test to restore afterwards."""
        return frozenset(self._values)

    def restore(self, values: frozenset[str]) -> None:
        """Put the registry back as `snapshot` found it.

        Process-wide state needs a way back for tests to stay independent of
        each other's credentials. Nothing in `src/` calls this.
        """
        self._values = set(values)
        self._ordered = tuple(sorted(self._values, key=len, reverse=True))


#: Process-wide, and deliberately not injected. Every logger is covered without
#: being handed anything — including the ones constructed inside services, which
#: have no composition root to ask. The failure modes decide it: this design
#: over-redacts when it is wrong, which is loud and harmless, where an injected
#: registry that a construction site forgot yields a logger indistinguishable
#: from a working one until the day it prints a key.
_REDACTIONS = _Redactions()


class Secret:
    """A credential. Renders as `PLACEHOLDER` everywhere except `expose_secret`."""

    __slots__ = ("_value",)

    def __init__(self, value: str) -> None:
        """Wrap a credential, registering it so it can never be logged."""
        self._value = value
        _REDACTIONS.register(value)

    def expose_secret(self) -> str:
        """The credential itself, for putting on the wire.

        Named to be conspicuous: it is the one deliberate act that takes a
        credential out of the type protecting it, and a structural guard
        allowlists every call site by name.
        """
        return self._value

    def __str__(self) -> str:
        return PLACEHOLDER

    __repr__ = __str__

    def __format__(self, format_spec: str) -> str:
        # The spec is ignored rather than applied: `format(self._value, spec)`
        # would hand the value straight back through an f-string.
        return PLACEHOLDER

    def __eq__(self, other: object) -> bool:
        # Only ever equal to another Secret. Comparing against a bare string
        # would make `==` an oracle: guess the value and the type confirms it.
        if not isinstance(other, Secret):
            return NotImplemented
        return self._value == other._value

    def __hash__(self) -> int:
        return hash(self._value)

    def __bool__(self) -> bool:
        return bool(self._value)


def scrub(text: str) -> str:
    """Replace every registered credential in `text` with the placeholder."""
    return _REDACTIONS.scrub(text)


def snapshot_redactions() -> frozenset[str]:
    """Registered values, for test isolation. Nothing in `src/` calls this."""
    return _REDACTIONS.snapshot()


def restore_redactions(values: frozenset[str]) -> None:
    """Restore the registry, for test isolation. Nothing in `src/` calls this."""
    _REDACTIONS.restore(values)
