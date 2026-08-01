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


# ---------------------------------------------------------------------------
# cosine
# ---------------------------------------------------------------------------


def test_cosine_identical_vectors_is_one():
    from src.utils.vectors import cosine

    v = [0.1, 0.2, 0.3, 0.4]
    assert cosine(v, v) == pytest.approx(1.0)


def test_cosine_orthogonal_vectors_is_zero():
    from src.utils.vectors import cosine

    assert cosine([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


def test_cosine_opposite_vectors_is_minus_one():
    from src.utils.vectors import cosine

    assert cosine([1.0, 2.0], [-1.0, -2.0]) == pytest.approx(-1.0)


def test_cosine_ignores_magnitude():
    """Cosine measures direction only — scaling either vector changes nothing."""
    from src.utils.vectors import cosine

    a = [1.0, 2.0, 3.0]
    scaled = [10.0, 20.0, 30.0]
    assert cosine(a, scaled) == pytest.approx(1.0)


def test_cosine_zero_vector_returns_zero():
    """A zero vector has no direction; must not raise ZeroDivisionError."""
    from src.utils.vectors import cosine

    assert cosine([0.0, 0.0], [1.0, 2.0]) == 0.0
    assert cosine([1.0, 2.0], [0.0, 0.0]) == 0.0
    assert cosine([0.0, 0.0], [0.0, 0.0]) == 0.0


def test_cosine_length_mismatch_raises():
    from src.utils.vectors import cosine

    with pytest.raises(ValueError, match="length"):
        cosine([1.0, 2.0], [1.0, 2.0, 3.0])


def test_cosine_known_value():
    from src.utils.vectors import cosine

    # 45 degrees between (1,0) and (1,1) -> cos = 1/sqrt(2)
    assert cosine([1.0, 0.0], [1.0, 1.0]) == pytest.approx(0.7071067811865475)
