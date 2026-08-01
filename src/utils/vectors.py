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


def pack_vectors(vectors: list[bytes]) -> bytes:
    """Concatenate encoded vectors into one BLOB.

    An event's tag embeddings share a single column, so they are stored
    end to end and split apart by count on the way out.

    Args:
        vectors: Encoded vectors, all of the same length.

    Returns:
        The concatenation, or empty bytes for an empty list.

    Raises:
        ValueError: If the vectors are not all the same length, which would
            make the stored BLOB impossible to split back apart.
    """
    if not vectors:
        return b""

    sizes = {len(v) for v in vectors}
    if len(sizes) > 1:
        raise ValueError(
            f"All vectors must be the same length to pack; got sizes {sorted(sizes)}"
        )

    return b"".join(vectors)


def unpack_vectors(blob: bytes, count: int) -> list[bytes]:
    """Split a packed BLOB back into individual encoded vectors.

    Args:
        blob: Bytes produced by pack_vectors.
        count: How many vectors it holds — typically the event's tag count.

    Returns:
        The individual encoded vectors.

    Raises:
        ValueError: If the blob does not divide evenly into count vectors.
    """
    if count <= 0:
        return []
    if len(blob) % count != 0:
        raise ValueError(
            f"Packed vectors ({len(blob)} bytes) do not divide evenly into {count}"
        )

    size = len(blob) // count
    return [blob[i * size : (i + 1) * size] for i in range(count)]


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
