"""Unit tests for the content-id rule handed to the adapters."""

from __future__ import annotations

from src.config import SourcesConfig
from src.ingestion.identity import content_id_rule


def test_an_unassigned_source_keeps_its_publisher_id():
    """`auto` means measure, not switch. Until a latch observes churn, the
    publisher's identifier is the better key — it survives an upstream edit,
    which a content-derived id does not."""
    rule = content_id_rule(SourcesConfig())

    assert rule("northshorenightout") is False


def test_a_source_assigned_to_content_derives_from_content():
    rule = content_id_rule(SourcesConfig(identity={"northshorenightout": "content"}))

    assert rule("northshorenightout") is True


def test_a_pinned_source_keeps_its_publisher_id():
    rule = content_id_rule(SourcesConfig(identity={"northshorenightout": "publisher"}))

    assert rule("northshorenightout") is False


def test_the_rule_is_per_source():
    """Assignments do not spread. A feed named here says nothing about its
    neighbours, including the other feed covering the same venues."""
    rule = content_id_rule(SourcesConfig(identity={"northshorenightout": "content"}))

    assert rule("northshorenightout_listing") is False
