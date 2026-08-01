"""Text normalisation for embedding inputs.

Some characters contribute nothing to an embedding but still cost signal.
Measured against `nomic-embed-text`: every emoji embeds to one identical
vector — `cosine(🍺, 🎤) == 1.000000` — which is also the vector produced by a
zero-width space, i.e. "nothing survived tokenization". An emoji therefore adds
no meaning of its own while diluting the words around it:

    '🎤 karaoke night' x 'karaoke' = 0.817
    'karaoke night'    x 'karaoke' = 0.861

Stripping them before embedding is strictly better. Applied to both preference
lines and event tags so the two sides of every comparison are normalised alike.
"""

from __future__ import annotations

import unicodedata

#: Unicode categories carrying no embedding signal:
#: Cf — format characters (zero-width space/joiner, BOM)
#: So — "symbol, other", which is where the bulk of emoji live
_ZERO_SIGNAL_CATEGORIES = frozenset({"Cf", "So"})

#: Ranges to drop that fall outside those categories. Targeted rather than
#: category-wide because their categories also contain characters that matter:
#: Mn holds combining accents (café) and Sk holds ^ and `.
_ZERO_SIGNAL_RANGES = (
    (0xFE00, 0xFE0F),    # variation selectors (emoji presentation)
    (0x1F3FB, 0x1F3FF),  # skin tone modifiers
)


def _is_zero_signal(char: str) -> bool:
    """True if the character contributes nothing to an embedding."""
    if unicodedata.category(char) in _ZERO_SIGNAL_CATEGORIES:
        return True
    code = ord(char)
    return any(low <= code <= high for low, high in _ZERO_SIGNAL_RANGES)


def _describe_symbols(text: str) -> str:
    """Render named symbols as their Unicode names, e.g. 🍺 -> 'beer mug'.

    A last resort for text that is nothing but symbols. The names are clumsy
    ('sign of the horns'), so this is never used when real words are present —
    measured, injecting 'microphone' into '🎤 karaoke night' *cost* 0.059
    similarity against 'karaoke'. But for a symbol on its own it beats the
    alternative: every emoji otherwise collapses to one identical vector whose
    nearest match is the same preference regardless of what it depicts.
    """
    names = []
    for char in text:
        if unicodedata.category(char) != "So":
            continue
        try:
            names.append(unicodedata.name(char).lower())
        except ValueError:
            continue  # unnamed codepoint — nothing to say about it
    return " ".join(names)


def normalize_embedding_text(text: str) -> str:
    """Strip zero-signal characters and collapse whitespace.

    Accents, currency symbols, punctuation, digits, and non-Latin scripts are
    preserved — only characters that vanish during tokenisation are removed.
    If stripping would leave nothing at all, named symbols are described
    instead so an emoji-only tag still carries some signal.

    Args:
        text: Raw tag or preference text.

    Returns:
        Normalised text. Empty only when nothing meaningful could be recovered.
    """
    stripped = " ".join(
        "".join(c for c in text if not _is_zero_signal(c)).split()
    )
    return stripped if stripped else _describe_symbols(text)
