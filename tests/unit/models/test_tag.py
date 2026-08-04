"""Tests for the weighted Tag model."""

from __future__ import annotations

import json

import pytest

from src.models.tag import Tag, tags_from_json, tags_to_json


def test_tag_holds_text_and_weight():
    tag = Tag(text="karaoke", weight=0.8)

    assert tag.text == "karaoke"
    assert tag.weight == 0.8


def test_weight_defaults_to_one():
    assert Tag(text="karaoke").weight == 1.0


@pytest.mark.parametrize("weight", [0.0, 0.5, 1.0])
def test_weight_accepts_full_range(weight):
    assert Tag(text="karaoke", weight=weight).weight == weight


@pytest.mark.parametrize("weight", [-0.1, 1.1, 2.0])
def test_weight_out_of_range_raises(weight):
    with pytest.raises(ValueError, match="weight"):
        Tag(text="karaoke", weight=weight)


def test_empty_text_raises():
    with pytest.raises(ValueError, match="text"):
        Tag(text="   ", weight=1.0)


def test_tags_compare_by_value():
    assert Tag(text="karaoke", weight=0.8) == Tag(text="karaoke", weight=0.8)
    assert Tag(text="karaoke", weight=0.8) != Tag(text="karaoke", weight=0.4)


# ---------------------------------------------------------------------------
# Serialisation — tags are persisted as JSON in events.tags
# ---------------------------------------------------------------------------


def test_tags_round_trip_through_json():

    tags = [Tag(text="karaoke", weight=1.0), Tag(text="bar", weight=0.2)]

    assert tags_from_json(tags_to_json(tags)) == tags


def test_serialised_form_carries_text_and_weight():


    payload = json.loads(tags_to_json([Tag(text="karaoke", weight=0.8)]))

    assert payload == [{"text": "karaoke", "weight": 0.8}]


def test_empty_tag_list_round_trips():

    assert tags_from_json(tags_to_json([])) == []


def test_legacy_bare_string_json_decodes_to_full_weight():
    """Rows written before weights existed must still load."""

    assert tags_from_json('["jazz", "live music"]') == [
        Tag(text="jazz", weight=1.0),
        Tag(text="live music", weight=1.0),
    ]
