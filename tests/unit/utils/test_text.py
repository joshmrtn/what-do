"""Unit tests for embedding text normalisation."""

from __future__ import annotations

import pytest

from src.utils.text import normalize_embedding_text


# ---------------------------------------------------------------------------
# Characters that carry no signal
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("🎤 karaoke", "karaoke"),
        ("karaoke 🎤", "karaoke"),
        ("🍻 beer 🍺 night", "beer night"),
        ("punk 🤘 rock", "punk rock"),
    ],
)
def test_emoji_stripped_from_mixed_text(raw, expected):
    """Emoji all embed to one constant vector, so they only dilute the tag."""
    assert normalize_embedding_text(raw) == expected


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("🎤", "microphone"),
        ("👻", "ghost"),
        ("💃", "dancer"),
        ("🍻🍺", "clinking beer mugs beer mug"),
    ],
)
def test_emoji_only_text_falls_back_to_unicode_names(raw, expected):
    """With no words to dilute, the name beats one constant vector for all emoji."""
    assert normalize_embedding_text(raw) == expected


def test_named_fallback_only_applies_when_nothing_else_remains():
    """Words present means the name would dilute rather than help."""
    assert normalize_embedding_text("🎤 karaoke") == "karaoke"


def test_zero_width_characters_stripped():
    assert normalize_embedding_text("kara​ok﻿e") == "karaoke"


def test_invisible_only_text_stays_empty():
    """Zero-width marks have no meaning to recover, unlike emoji."""
    assert normalize_embedding_text("​﻿‍") == ""


def test_variation_selectors_and_skin_tones_stripped():
    assert normalize_embedding_text("wave \U0001F44B\U0001F3FD️ now") == "wave now"


def test_whitespace_collapsed_and_trimmed():
    assert normalize_embedding_text("  live    music  ") == "live music"


def test_empty_input_returns_empty():
    assert normalize_embedding_text("") == ""
    assert normalize_embedding_text("   ") == ""


# ---------------------------------------------------------------------------
# Characters that DO carry signal — must survive
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "café",            # combining/precomposed accents
        "R&B",             # ampersand
        "$5 shows",        # currency
        "rock 'n' roll",   # apostrophes
        "post-punk",       # hyphen
        "hip-hop / rap",   # slash
        "90s night",       # digits
        "караоке",         # non-latin script
        "drum & bass!",    # punctuation
    ],
)
def test_meaningful_text_preserved(text):
    assert normalize_embedding_text(text) == text
