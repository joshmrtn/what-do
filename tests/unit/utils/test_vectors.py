import math

import pytest

from src.utils.vectors import cosine, decode_vector, encode_vector, pack_vectors, unpack_vectors


def test_encode_returns_bytes():

    result = encode_vector([0.1, 0.2, 0.3])
    assert isinstance(result, bytes)


def test_encode_uses_float32_storage():

    v = [1.0] * 768
    assert len(encode_vector(v)) == 768 * 4  # 4 bytes per float32


def test_roundtrip_lossless_768_dims():

    v = [float(i) * 0.001 for i in range(768)]
    result = decode_vector(encode_vector(v))

    assert len(result) == 768
    for orig, decoded in zip(v, result):
        assert math.isclose(orig, decoded, rel_tol=1e-5), (
            f"Precision loss: {orig} → {decoded}"
        )


def test_decode_roundtrip_small_vector():

    v = [1.5, -0.5, 0.0, 3.14]
    assert decode_vector(encode_vector(v)) == pytest.approx(v, rel=1e-5)


def test_empty_vector_roundtrip():

    assert decode_vector(encode_vector([])) == []


# ---------------------------------------------------------------------------
# cosine
# ---------------------------------------------------------------------------


def test_cosine_identical_vectors_is_one():

    v = [0.1, 0.2, 0.3, 0.4]
    assert cosine(v, v) == pytest.approx(1.0)


def test_cosine_orthogonal_vectors_is_zero():

    assert cosine([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


def test_cosine_opposite_vectors_is_minus_one():

    assert cosine([1.0, 2.0], [-1.0, -2.0]) == pytest.approx(-1.0)


def test_cosine_ignores_magnitude():
    """Cosine measures direction only — scaling either vector changes nothing."""

    a = [1.0, 2.0, 3.0]
    scaled = [10.0, 20.0, 30.0]
    assert cosine(a, scaled) == pytest.approx(1.0)


def test_cosine_zero_vector_returns_zero():
    """A zero vector has no direction; must not raise ZeroDivisionError."""

    assert cosine([0.0, 0.0], [1.0, 2.0]) == 0.0
    assert cosine([1.0, 2.0], [0.0, 0.0]) == 0.0
    assert cosine([0.0, 0.0], [0.0, 0.0]) == 0.0


def test_cosine_length_mismatch_raises():

    with pytest.raises(ValueError, match="length"):
        cosine([1.0, 2.0], [1.0, 2.0, 3.0])


def test_cosine_known_value():

    # 45 degrees between (1,0) and (1,1) -> cos = 1/sqrt(2)
    assert cosine([1.0, 0.0], [1.0, 1.0]) == pytest.approx(0.7071067811865475)


# ---------------------------------------------------------------------------
# pack_vectors / unpack_vectors — many vectors in one BLOB column
# ---------------------------------------------------------------------------


def test_pack_concatenates_encoded_vectors():

    a, b = encode_vector([1.0, 2.0]), encode_vector([3.0, 4.0])

    assert pack_vectors([a, b]) == a + b


def test_pack_empty_list_is_empty_bytes():

    assert pack_vectors([]) == b""


def test_unpack_round_trips():

    vectors = [encode_vector([float(i), float(i + 1), float(i + 2)]) for i in range(4)]

    assert unpack_vectors(pack_vectors(vectors), count=4) == vectors


def test_unpack_round_trip_preserves_values_at_768_dims():

    originals = [[i * 0.001 for i in range(768)], [i * 0.002 for i in range(768)]]
    packed = pack_vectors([encode_vector(v) for v in originals])

    restored = [decode_vector(b) for b in unpack_vectors(packed, count=2)]

    assert restored[0] == pytest.approx(originals[0])
    assert restored[1] == pytest.approx(originals[1])


def test_unpack_zero_count_returns_empty():

    assert unpack_vectors(b"", count=0) == []


def test_unpack_uneven_split_raises():

    with pytest.raises(ValueError, match="evenly"):
        unpack_vectors(b"\x00" * 10, count=3)


def test_pack_rejects_ragged_vectors():
    """Unpacking splits evenly, so unequal lengths could not be recovered."""

    with pytest.raises(ValueError, match="same length"):
        pack_vectors([encode_vector([1.0, 2.0]), encode_vector([3.0])])
