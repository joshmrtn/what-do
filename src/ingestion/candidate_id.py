"""Stable candidate id derivation, shared by the ingestion adapters."""

from __future__ import annotations

import hashlib
from datetime import datetime

_DIGEST_LENGTH = 16


def derive_candidate_id(source_type: str, *parts: object) -> str:
    """Derive a candidate id that is identical on every fetch of the same item.

    Prefer the source's own identifier as the material. Content is a fallback for
    sources that publish none: it is stable across fetches, but an edit upstream
    mints a new id and re-extracts, which an identifier would not.

    Args:
        source_type: The adapter's source_type, used as a readable prefix.
        *parts: Material identifying the upstream item. `None` holds its position
            so an absent field cannot shift the remaining values into it.

    Returns:
        An id of the form ``"<source_type>:<16 hex chars>"``.

    Raises:
        ValueError: If no part carries content, which would otherwise collapse
            every such item onto one id.
    """
    rendered = [_render(part) for part in parts]
    if not any(rendered):
        raise ValueError(f"no identifying material for a {source_type} candidate")
    digest = hashlib.sha256("|".join(rendered).encode("utf-8")).hexdigest()
    return f"{source_type}:{digest[:_DIGEST_LENGTH]}"


def _render(part: object) -> str:
    if part is None:
        return ""
    if isinstance(part, datetime):
        return part.isoformat()
    return str(part)
