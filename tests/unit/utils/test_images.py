"""Identifying an image by its bytes rather than by what it was called."""

from __future__ import annotations

from src.utils.images import DEFAULT_IMAGE_MIME, sniff_mime_type

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 16
JPEG = b"\xff\xd8\xff\xe0" + b"\x00" * 16
GIF87 = b"GIF87a" + b"\x00" * 16
GIF89 = b"GIF89a" + b"\x00" * 16
WEBP = b"RIFF" + b"\x24\x00\x00\x00" + b"WEBP" + b"\x00" * 16


class TestSniffMimeType:
    def test_png(self):
        assert sniff_mime_type(PNG) == "image/png"

    def test_jpeg(self):
        assert sniff_mime_type(JPEG) == "image/jpeg"

    def test_gif_87a(self):
        assert sniff_mime_type(GIF87) == "image/gif"

    def test_gif_89a(self):
        assert sniff_mime_type(GIF89) == "image/gif"

    def test_webp(self):
        """RIFF is a container marker, so the format lives four bytes later —
        reading only the prefix would call every RIFF file a WebP."""
        assert sniff_mime_type(WEBP) == "image/webp"

    def test_a_riff_container_that_is_not_webp_is_not_claimed(self):
        wav = b"RIFF" + b"\x24\x00\x00\x00" + b"WAVE" + b"\x00" * 16

        assert sniff_mime_type(wav) == DEFAULT_IMAGE_MIME

    def test_unrecognised_bytes_fall_back_rather_than_raising(self):
        """The fallback is what the client hardcoded before, so an unknown
        format is no worse off than it was — and a fetch that reached this far
        should not die over a label."""
        assert sniff_mime_type(b"not an image at all") == DEFAULT_IMAGE_MIME

    def test_empty_bytes_fall_back(self):
        assert sniff_mime_type(b"") == DEFAULT_IMAGE_MIME

    def test_bytes_shorter_than_a_signature_do_not_raise(self):
        assert sniff_mime_type(b"\x89PN") == DEFAULT_IMAGE_MIME
