"""Unit tests for the short display handle."""

from src.presentation.handles import short_handle


def test_a_handle_is_seven_lowercase_hex_characters():
    """Short enough to sit in a listing, long enough that 1935 events collide
    with probability under a percent."""
    handle = short_handle("531a3fa7-8515-44d7-8ac4-c2f6180a1a10")

    assert len(handle) == 7
    assert all(character in "0123456789abcdef" for character in handle)


def test_the_same_id_always_gives_the_same_handle():
    """Event ids are stable across runs — measured, 530 events created 08-09
    were still ranked under the same id on 08-16 — so a handle copied off
    yesterday's listing has to resolve today."""
    first = short_handle("531a3fa7-8515-44d7-8ac4-c2f6180a1a10")
    second = short_handle("531a3fa7-8515-44d7-8ac4-c2f6180a1a10")

    assert first == second


def test_two_ids_give_two_handles():
    first = short_handle("531a3fa7-8515-44d7-8ac4-c2f6180a1a10")
    second = short_handle("b043ea88-e52c-4159-b279-962b812e994e")

    assert first != second


def test_a_non_uuid_id_still_gives_seven_hex_characters():
    """Ids are not uniformly UUIDs. Synthetic events are keyed on their rule and
    date, so a slice of the id would read `synthet` for every one of them —
    which is why the handle is derived rather than taken."""
    handle = short_handle("synthetic:evening_walk:2026-08-16")

    assert len(handle) == 7
    assert all(character in "0123456789abcdef" for character in handle)
    assert not handle.startswith("synthet")


def test_synthetic_ids_that_differ_only_by_date_give_different_handles():
    """Every night's evening walk is its own event and must be addressable as
    one — the case that motivated a handle in the first place."""
    handles = {
        short_handle(f"synthetic:evening_walk:2026-08-{day}")
        for day in ("09", "11", "15", "16")
    }

    assert len(handles) == 4
