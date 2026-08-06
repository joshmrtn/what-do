"""HTML-to-text conversion for source descriptions.

Some sources embed light markup in otherwise plain-text fields. LLM Pass 1
copes with messy input, but it cannot recover information it never receives, so
this converts rather than strips: block-level breaks become newlines instead of
disappearing, and a link's href is kept alongside its text.

Deleting a tag outright is the subtle failure — `Boston!<br>You will` becomes
`Boston!You will`, inventing a word that was never in the source.
"""

from __future__ import annotations

import html as _html
import re

#: Tags that represent a line break rather than inline emphasis.
_BREAK_TAGS = ("br", "p", "div", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6")

_BREAK_RE = re.compile(
    r"<\s*/?\s*(?:" + "|".join(_BREAK_TAGS) + r")\b[^>]*>",
    re.IGNORECASE,
)
_ANCHOR_RE = re.compile(
    r"<\s*a\b[^>]*?href\s*=\s*(\"[^\"]*\"|'[^']*')[^>]*>(.*?)<\s*/\s*a\s*>",
    re.IGNORECASE | re.DOTALL,
)
_TAG_RE = re.compile(r"<[^>]+>")
_BLANK_RUN_RE = re.compile(r"\n{3,}")


def _expand_anchor(match: re.Match[str]) -> str:
    """Render an anchor as `text (url)`, keeping the href out of the bin."""
    href = match.group(1)[1:-1].strip()
    text = _TAG_RE.sub("", match.group(2)).strip()

    if not href:
        return text
    if not text or text == href:
        return href
    return f"{text} ({href})"


def html_to_text(value: str) -> str:
    """Convert light HTML markup to plain text without losing content.

    Args:
        value: Source text, which may or may not contain markup.

    Returns:
        Plain text with breaks preserved as newlines, link targets retained,
        and HTML entities resolved.
    """
    if not value:
        return value

    if "<" not in value and "&" not in value and "\xa0" not in value:
        return value

    text = _ANCHOR_RE.sub(_expand_anchor, value)
    text = _BREAK_RE.sub("\n", text)
    text = _TAG_RE.sub("", text)
    text = _html.unescape(text)

    # A non-breaking space is a space, not a character worth carrying forward.
    text = text.replace("\xa0", " ")

    text = _BLANK_RUN_RE.sub("\n\n", text)
    return text.strip()
