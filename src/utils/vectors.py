"""Utilities for encoding, decoding, and comparing embedding vectors.

Vectors are persisted as float32 BLOBs and compared with cosine similarity.
"""

from __future__ import annotations

import math
import struct


def encode_vector(v: list[float]) -> bytes:
    """Encode a float list as a float32 binary BLOB.

    Args:
        v: List of floats to encode.

    Returns:
        Packed bytes in float32 format.
    """
    return struct.pack(f"{len(v)}f", *v)


def decode_vector(b: bytes) -> list[float]:
    """Decode a float32 binary BLOB back to a float list.

    Args:
        b: Bytes produced by encode_vector.

    Returns:
        List of floats.
    """
    count = len(b) // 4
    return list(struct.unpack(f"{count}f", b))


def cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two equal-length vectors.

    Measures direction only — magnitude is normalised away.

    Args:
        a: First vector.
        b: Second vector, same length as a.

    Returns:
        Similarity in [-1.0, 1.0]. Returns 0.0 if either vector has zero
        magnitude, since a zero vector has no direction to compare.

    Raises:
        ValueError: If the vectors differ in length.
    """
    if len(a) != len(b):
        raise ValueError(
            f"Cannot compare vectors of different length: {len(a)} != {len(b)}"
        )

    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for x, y in zip(a, b):
        dot += x * y
        norm_a += x * x
        norm_b += y * y

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    return dot / (math.sqrt(norm_a) * math.sqrt(norm_b))
