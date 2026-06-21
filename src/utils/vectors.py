"""Utilities for encoding and decoding embedding vectors as float32 BLOBs."""

from __future__ import annotations

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
