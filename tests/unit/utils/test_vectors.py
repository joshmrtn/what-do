import math

import pytest


def test_encode_returns_bytes():
    from src.utils.vectors import encode_vector

    result = encode_vector([0.1, 0.2, 0.3])
    assert isinstance(result, bytes)


def test_encode_uses_float32_storage():
    from src.utils.vectors import encode_vector

    v = [1.0] * 768
    assert len(encode_vector(v)) == 768 * 4  # 4 bytes per float32


def test_roundtrip_lossless_768_dims():
    from src.utils.vectors import decode_vector, encode_vector

    v = [float(i) * 0.001 for i in range(768)]
    result = decode_vector(encode_vector(v))

    assert len(result) == 768
    for orig, decoded in zip(v, result):
        assert math.isclose(orig, decoded, rel_tol=1e-5), (
            f"Precision loss: {orig} → {decoded}"
        )


def test_decode_roundtrip_small_vector():
    from src.utils.vectors import decode_vector, encode_vector

    v = [1.5, -0.5, 0.0, 3.14]
    assert decode_vector(encode_vector(v)) == pytest.approx(v, rel=1e-5)


def test_empty_vector_roundtrip():
    from src.utils.vectors import decode_vector, encode_vector

    assert decode_vector(encode_vector([])) == []
