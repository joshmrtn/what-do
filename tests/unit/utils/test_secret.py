"""A credential is a type that cannot be printed and a value that cannot be logged.

Two layers, tested apart because they fail apart. `Secret` stops a credential
being *rendered* — an f-string, a repr, a dataclass printing its fields — and it
works for any value, however short. `scrub` stops a credential that has already
been handed to `expose_secret()` and is now sitting inside text somebody else
built, which is the only way it can reach a log at all once the type is in
place.

The floor belongs to the second layer only. A very short credential is still
unprintable; it is merely not searched for in arbitrary text, because a
three-character value would redact its own initials out of every line in the
batch.
"""

from __future__ import annotations

import io
import json

import pytest

from src.utils.logging import get_logger
from src.utils.secret import MIN_REDACTABLE_LENGTH, PLACEHOLDER, Secret, scrub

#: Long enough to be registered, and shaped like the thing it stands in for.
SENTINEL = "tmdb-sentinel-9f3a2c7e5b"


def test_str_does_not_reveal():
    assert str(Secret(SENTINEL)) == PLACEHOLDER


def test_repr_does_not_reveal():
    """The one that leaks by accident: a dataclass prints its fields with repr."""
    assert repr(Secret(SENTINEL)) == PLACEHOLDER
    assert SENTINEL not in repr([Secret(SENTINEL)])
    assert SENTINEL not in repr({"api_key": Secret(SENTINEL)})


def test_interpolation_does_not_reveal():
    secret = Secret(SENTINEL)

    assert f"Bearer {secret}" == f"Bearer {PLACEHOLDER}"
    assert "Bearer %s" % secret == f"Bearer {PLACEHOLDER}"
    assert "Bearer {}".format(secret) == f"Bearer {PLACEHOLDER}"


def test_a_format_spec_cannot_widen_its_way_to_the_value():
    """`__format__` ignores the spec rather than falling back to the value."""
    assert SENTINEL not in f"{Secret(SENTINEL):>60}"
    assert SENTINEL not in f"{Secret(SENTINEL):.100}"


def test_expose_secret_returns_the_value():
    assert Secret(SENTINEL).expose_secret() == SENTINEL


def test_equality_compares_values():
    """`AppConfig` is a dataclass, so identity comparison would break config equality."""
    assert Secret(SENTINEL) == Secret(SENTINEL)
    assert Secret(SENTINEL) != Secret("some-other-credential")


def test_equality_against_a_bare_string_is_false():
    """Otherwise `==` is an oracle: guess the value, and the type confirms it."""
    assert Secret(SENTINEL) != SENTINEL
    assert SENTINEL != Secret(SENTINEL)


def test_hash_agrees_with_equality():
    assert len({Secret(SENTINEL), Secret(SENTINEL)}) == 1
    assert {Secret(SENTINEL): "tmdb"}[Secret(SENTINEL)] == "tmdb"


def test_json_refuses_a_secret():
    """Serialising one is not something to make convenient."""
    with pytest.raises(TypeError):
        json.dumps({"api_key": Secret(SENTINEL)})


def test_scrub_replaces_a_registered_value_inside_surrounding_text():
    Secret(SENTINEL)

    scrubbed = scrub(f"401 Client Error for url: https://api/search?api_key={SENTINEL}&q=dune")

    assert SENTINEL not in scrubbed
    assert PLACEHOLDER in scrubbed
    # The line still reads: everything around the value survives.
    assert scrubbed.startswith("401 Client Error for url: https://api/search?api_key=")
    assert scrubbed.endswith("&q=dune")


def test_scrub_leaves_text_holding_no_secret_alone():
    Secret(SENTINEL)

    message = "tmdb: api.themoviedb.org failed after 3 attempt(s)"

    assert scrub(message) == message


def test_scrub_replaces_every_occurrence():
    Secret(SENTINEL)

    assert SENTINEL not in scrub(f"{SENTINEL} retried, then {SENTINEL} again")


def test_a_value_under_the_floor_is_never_registered():
    """A short value would redact its own letters out of every unrelated line."""
    short = "a" * (MIN_REDACTABLE_LENGTH - 1)
    Secret(short)

    assert scrub(f"a banana: {short}") == f"a banana: {short}"


def test_a_value_under_the_floor_is_still_unprintable():
    """The floor belongs to the backstop. The type protects every value."""
    short = "a" * (MIN_REDACTABLE_LENGTH - 1)

    assert str(Secret(short)) == PLACEHOLDER
    assert Secret(short).expose_secret() == short


def test_a_secret_containing_another_is_replaced_whole():
    """Shortest-first would replace the inner value and strand the outer's tail."""
    inner = "sentinel-abcdefgh"
    outer = f"{inner}-and-then-some"
    Secret(inner)
    Secret(outer)

    scrubbed = scrub(f"token={outer}")

    assert scrubbed == f"token={PLACEHOLDER}"
    assert "and-then-some" not in scrubbed


def test_a_logged_message_carrying_an_exposed_secret_is_scrubbed():
    """The whole point: one hook, in the one place every log line is rendered."""
    secret = Secret(SENTINEL)
    stream = io.StringIO()
    log = get_logger("test.secret.scrubbed", stream=stream)

    log.error(
        f"tmdb: failed for url https://api/search?api_key={secret.expose_secret()}",
        component="network",
        duration_ms=0,
    )

    stream.seek(0)
    entry = json.loads(stream.readline())
    assert SENTINEL not in entry["message"]
    assert PLACEHOLDER in entry["message"]


def test_scrubbing_survives_a_logger_built_without_ceremony():
    """A logger nobody handed a redactor to still redacts, which is why it is central."""
    secret = Secret(SENTINEL)
    stream = io.StringIO()

    get_logger("test.secret.plain", stream=stream).info(
        secret.expose_secret(), component="", duration_ms=0
    )

    stream.seek(0)
    assert SENTINEL not in stream.read()
