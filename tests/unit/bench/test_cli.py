"""Unit tests for the bench CLI's own logic.

Everything up to the model call is ours and is tested; the model call is the
seam and is not. That is the same line the pipeline draws.
"""

from __future__ import annotations

import pytest

from src.bench.cli import BenchError, build_variants, parse_args


def test_a_variant_is_named_for_its_model_by_default():
    variants = build_variants(["gemma4:e4b", "gemma4:e2b"], host="http://localhost:11434")

    assert [v.name for v in variants] == ["gemma4:e4b", "gemma4:e2b"]
    assert [v.model for v in variants] == ["gemma4:e4b", "gemma4:e2b"]


def test_a_variant_may_be_named_apart_from_its_model():
    """Two variants of the *same* model differing in prompt or input shape is
    the comparison the bench was built for, and they need distinguishable
    names."""
    variants = build_variants(["venue-line=gemma4:e4b"], host="http://localhost:11434")

    assert variants[0].name == "venue-line"
    assert variants[0].model == "gemma4:e4b"


def test_at_least_one_variant_is_required():
    with pytest.raises(BenchError, match="variant"):
        build_variants([], host="http://localhost:11434")


def test_duplicate_variant_names_are_rejected():
    """Two columns with one name makes the table unreadable and the recorded
    baseline ambiguous."""
    with pytest.raises(BenchError, match="twice"):
        build_variants(["gemma4:e4b", "gemma4:e4b"], host="http://localhost:11434")


def test_from_db_ids_are_split():
    args = parse_args(["extraction", "--variant", "x", "--from-db", "a,b,c"])

    assert args.from_db == ["a", "b", "c"]


def test_from_db_is_empty_when_absent():
    args = parse_args(["extraction", "--variant", "x"])

    assert args.from_db == []


def test_the_subcommand_is_required():
    with pytest.raises(SystemExit):
        parse_args([])
