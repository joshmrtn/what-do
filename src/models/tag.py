"""Weighted event tag.

Each tag carries a centrality weight: how central the tag is to what the event
actually is. The scoring layer multiplies a tag's similarity contribution by this
weight, so incidental attributes (the kind of venue, the day of week) recede
relative to the main activity.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

MIN_WEIGHT = 0.0
MAX_WEIGHT = 1.0
DEFAULT_WEIGHT = 1.0


@dataclass(frozen=True)
class Tag:
    """A descriptive event tag with a centrality weight in [0.0, 1.0].

    Args:
        text: The tag itself, e.g. "karaoke".
        weight: How central the tag is to the event. 1.0 = defining feature,
            0.5 = secondary attribute, 0.1 = incidental context.

    Raises:
        ValueError: If text is blank or weight falls outside [0.0, 1.0].
    """

    text: str
    weight: float = DEFAULT_WEIGHT

    def __post_init__(self) -> None:
        if not self.text or not self.text.strip():
            raise ValueError("Tag text must be a non-empty string")
        if not MIN_WEIGHT <= self.weight <= MAX_WEIGHT:
            raise ValueError(
                f"Tag weight {self.weight} out of range "
                f"[{MIN_WEIGHT}, {MAX_WEIGHT}] for tag {self.text!r}"
            )


def clamp_weight(value: object) -> float:
    """Coerce an untrusted weight into [0.0, 1.0], defaulting when unusable.

    LLM output is not schema-guaranteed, so a missing, non-numeric, or
    out-of-range weight degrades to a usable value rather than failing
    extraction. Booleans are rejected — ``True`` is numerically 1.0 but is
    never a weight the model meant to emit.

    Args:
        value: Raw weight from parsed model output.

    Returns:
        The weight clamped to range, or DEFAULT_WEIGHT if not a usable number.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return DEFAULT_WEIGHT
    return max(MIN_WEIGHT, min(MAX_WEIGHT, float(value)))


def tags_to_json(tags: list[Tag]) -> str:
    """Serialise tags for the events.tags column."""
    return json.dumps([{"text": t.text, "weight": t.weight} for t in tags])


def tags_from_json(raw: str) -> list[Tag]:
    """Deserialise the events.tags column.

    Accepts both the weighted form and a bare list of strings, so rows written
    before weights existed still load at DEFAULT_WEIGHT.

    Args:
        raw: JSON text from the events.tags column.

    Returns:
        Decoded tags, skipping entries with no usable text.
    """
    data = json.loads(raw)
    tags: list[Tag] = []
    for entry in data:
        if isinstance(entry, str):
            text, weight = entry, DEFAULT_WEIGHT
        else:
            text = entry.get("text", "")
            weight = clamp_weight(entry.get("weight"))
        if text and text.strip():
            tags.append(Tag(text=text, weight=weight))
    return tags
