"""Test isolation for state that is deliberately process-wide.

The credential registry in `src/utils/secret.py` is not injected, for reasons
written there. The cost of that choice is exactly this fixture: a `Secret`
minted by one test would otherwise still be scrubbed from another's log lines,
making a passing test depend on what ran before it.

The isolation runs in the safe direction either way — a leftover registration
over-redacts, which shows up as a test seeing the placeholder where it expected
text. It never lets a credential through.
"""

from __future__ import annotations

from typing import Iterator

import pytest

from src.utils.secret import restore_redactions, snapshot_redactions


@pytest.fixture(autouse=True)
def isolate_credential_registry() -> Iterator[None]:
    """Restore the credential registry to whatever the test found it holding."""
    before = snapshot_redactions()
    yield
    restore_redactions(before)
