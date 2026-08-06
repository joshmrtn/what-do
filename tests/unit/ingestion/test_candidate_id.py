"""Unit tests for stable candidate id derivation."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from src.ingestion.candidate_id import derive_candidate_id


def test_same_material_yields_same_id():
    """The whole point: a refetch of the same item resolves to the same id."""
    first = derive_candidate_id("apify", "post_abc123")
    second = derive_candidate_id("apify", "post_abc123")
    assert first == second


def test_different_material_yields_different_id():
    assert derive_candidate_id("apify", "post_abc") != derive_candidate_id("apify", "post_xyz")


def test_source_type_prefixes_the_id():
    assert derive_candidate_id("apify", "post_abc").startswith("apify:")


def test_same_material_under_different_source_types_differs():
    """Two sources describing one upstream item must not collide on id."""
    assert derive_candidate_id("picuki", "abc") != derive_candidate_id("dumpor", "abc")


def test_composite_material_is_order_sensitive():
    """A film id and a showtime are not interchangeable."""
    assert derive_candidate_id("v", "film_1", "20:30") != derive_candidate_id("v", "20:30", "film_1")


def test_none_parts_are_tolerated():
    derived = derive_candidate_id("apify", None, "post_abc")
    assert derived.startswith("apify:")


def test_none_is_distinct_from_empty_string_position():
    """A missing part must not silently shift the remaining material."""
    assert derive_candidate_id("v", None, "b") != derive_candidate_id("v", "b", None)


def test_datetime_material_is_accepted():
    when = datetime(2025, 6, 16, 20, 30, tzinfo=timezone.utc)
    assert derive_candidate_id("v", "film_1", when) == derive_candidate_id("v", "film_1", when)


def test_id_is_fixed_length_regardless_of_material_size():
    short = derive_candidate_id("apify", "a")
    long = derive_candidate_id("apify", "a" * 5000)
    assert len(short) == len(long)


def test_rejects_material_with_no_content():
    """All-empty material would collapse every such item onto one id."""
    with pytest.raises(ValueError):
        derive_candidate_id("apify", None, "")
