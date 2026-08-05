"""Tests for the shared blocklist matcher."""

import pytest

from src.utils.blocklist import is_blocked

THRESHOLD = 0.80


def test_exact_handle_blocks():
    assert is_blocked("Some Bar", ["@sportsbar"], ["@sportsbar"], THRESHOLD)


def test_different_handle_does_not_block():
    assert not is_blocked("Some Bar", ["@jazzclub"], ["@sportsbar"], THRESHOLD)


def test_handle_entry_never_matches_a_name():
    """An @handle entry is an identity, not a fuzzy string to compare against names."""
    assert not is_blocked("sportsbar", [], ["@sportsbar"], THRESHOLD)


def test_identical_name_blocks():
    assert is_blocked("The Sports Bar", [], ["The Sports Bar"], THRESHOLD)


def test_name_match_is_case_insensitive():
    assert is_blocked("the sports bar", [], ["The Sports Bar"], THRESHOLD)


def test_near_miss_above_threshold_blocks():
    assert is_blocked("The Sports Bar", [], ["The Sport Bar"], THRESHOLD)


def test_unrelated_name_does_not_block():
    assert not is_blocked("Jazz Cellar", [], ["The Sports Bar"], THRESHOLD)


def test_threshold_decides_a_borderline_match():
    """The same pair blocks under a lenient threshold and passes under a strict one."""
    name, entry = "The Sports Bar", "Sports Bar and Grill"

    assert is_blocked(name, [], [entry], 0.5)
    assert not is_blocked(name, [], [entry], 0.99)


def test_empty_blocklist_blocks_nothing():
    assert not is_blocked("The Sports Bar", ["@sportsbar"], [], THRESHOLD)


def test_missing_name_is_never_blocked_by_a_name_entry():
    """An event with no venue has nothing to compare; it must not match by accident."""
    assert not is_blocked(None, [], ["The Sports Bar"], THRESHOLD)


def test_missing_name_can_still_be_blocked_by_handle():
    assert is_blocked(None, ["@sportsbar"], ["@sportsbar"], THRESHOLD)


@pytest.mark.parametrize("blank", ["", "   "])
def test_blank_name_is_never_blocked(blank):
    assert not is_blocked(blank, [], ["The Sports Bar"], THRESHOLD)


def test_any_matching_entry_blocks():
    assert is_blocked("Jazz Cellar", [], ["The Sports Bar", "Jazz Cellar"], THRESHOLD)
