"""Tests for turning a loaded preference set into a recordable revision.

The revision answers one question: *were these the preferences that produced
that ranking?* So the hash covers what actually scores — the lines, their order,
their domains — and nothing else.
"""

from __future__ import annotations

from datetime import datetime, timezone

from src.scoring.preference_revision import build_revision
from src.scoring.preferences import PreferenceSet, UserPreference

CAPTURED = datetime(2026, 8, 16, 2, 0, tzinfo=timezone.utc)


def _set(likes: list[str], dislikes: list[str], domain: str = "general") -> PreferenceSet:
    return PreferenceSet(
        likes=[UserPreference("like", domain, text) for text in likes],
        dislikes=[UserPreference("dislike", domain, text) for text in dislikes],
    )


def _build(preferences: PreferenceSet):
    return build_revision(
        preferences,
        likes_name="likes.txt",
        dislikes_name="dislikes.txt",
        captured_at=CAPTURED,
    )


def test_the_same_preferences_hash_the_same():
    """Two batches over an unedited file must reuse one revision."""
    first = _build(_set(["live music"], ["karaoke"]))
    second = _build(_set(["live music"], ["karaoke"]))

    assert first.content_hash == second.content_hash


def test_changing_a_line_changes_the_hash():
    assert (
        _build(_set(["live music"], ["karaoke"])).content_hash
        != _build(_set(["live jazz"], ["karaoke"])).content_hash
    )


def test_a_like_and_a_dislike_of_the_same_text_differ():
    """The text alone is not the preference — which list it is on is the point."""
    assert (
        _build(_set(["karaoke"], [])).content_hash
        != _build(_set([], ["karaoke"])).content_hash
    )


def test_reordering_lines_changes_the_hash():
    """Position is recorded, so a reorder is a real revision.

    Scoring takes a max over the list, so an order change does not move a score
    today — but the revision describes the file, and claiming a reordered file
    is the same one would make the record lie about something it stores.
    """
    assert (
        _build(_set(["live music", "trivia"], [])).content_hash
        != _build(_set(["trivia", "live music"], [])).content_hash
    )


def test_the_domain_is_part_of_the_revision():
    """A line moved under `[movies]` applies to different events entirely."""
    assert (
        _build(_set(["subtitles"], [], domain="general")).content_hash
        != _build(_set(["subtitles"], [], domain="movies")).content_hash
    )


def test_embeddings_do_not_reach_the_hash():
    """A revision describes the file, not the cache.

    Vectors are a pure function of text and model, so folding them in would mint
    a new revision every time the embedding model changed while the user's
    preferences stood still.
    """
    plain = _set(["live music"], [])
    embedded = _set(["live music"], [])
    embedded.likes[0].embedding = [0.1, 0.2, 0.3]

    assert _build(plain).content_hash == _build(embedded).content_hash


def test_lines_carry_their_file_position_domain_and_type():
    revision = _build(_set(["live music", "trivia"], ["karaoke"], domain="movies"))

    likes = [line for line in revision.lines if line.preference_type == "like"]
    dislikes = [line for line in revision.lines if line.preference_type == "dislike"]

    assert [(line.file_name, line.position, line.line_text) for line in likes] == [
        ("likes.txt", 0, "live music"),
        ("likes.txt", 1, "trivia"),
    ]
    assert [(line.file_name, line.position, line.line_text) for line in dislikes] == [
        ("dislikes.txt", 0, "karaoke"),
    ]
    assert {line.domain for line in revision.lines} == {"movies"}


def test_every_line_carries_a_hash_of_its_own_text():
    """The same key the embedding cache uses, so a line can be joined to it."""
    revision = _build(_set(["live music"], []))
    line = revision.lines[0]

    assert line.line_hash == build_revision(
        _set(["live music"], []),
        likes_name="other.txt",
        dislikes_name="dislikes.txt",
        captured_at=CAPTURED,
    ).lines[0].line_hash


def test_an_empty_preference_set_still_has_a_hash():
    """A user with no preferences yet is a real state, not an error."""
    revision = _build(_set([], []))

    assert revision.content_hash
    assert revision.lines == []
