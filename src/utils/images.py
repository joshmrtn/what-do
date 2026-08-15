"""Identifying an image by its bytes."""

from __future__ import annotations

#: What an unrecognised image is labelled. Not a guess dressed as knowledge —
#: it is what the Gemini client hardcoded for *every* image before this existed,
#: so an unknown format is no worse off than it was, and a fetch that got this
#: far should not fail over a label.
DEFAULT_IMAGE_MIME = "image/jpeg"

#: Leading signatures, longest first so a shorter prefix cannot claim a file
#: that a longer one identifies precisely.
_SIGNATURES: tuple[tuple[bytes, str], ...] = (
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"GIF87a", "image/gif"),
    (b"GIF89a", "image/gif"),
)

#: WebP is a RIFF container, so its format marker sits after the four-byte size
#: field rather than at the start. Matching on `RIFF` alone would call a WAV an
#: image.
_RIFF = b"RIFF"
_WEBP_AT = 8


def sniff_mime_type(image: bytes) -> str:
    """The image's real media type, read from its own leading bytes.

    Sniffed rather than taken from the HTTP `Content-Type`, for two reasons:
    servers mislabel images routinely, and `ImageFetcher.fetch` returns bare
    bytes — threading a header through would change the protocol and every
    implementation of it to carry something less trustworthy than what is
    already in hand.

    Args:
        image: Raw image bytes, however few.

    Returns:
        The matched media type, or `DEFAULT_IMAGE_MIME` when nothing matches.
        Short or empty input falls back rather than raising: this is called on
        the way to a model, and the caller has already paid for the fetch.
    """
    for signature, mime in _SIGNATURES:
        if image.startswith(signature):
            return mime

    if image.startswith(_RIFF) and image[_WEBP_AT : _WEBP_AT + 4] == b"WEBP":
        return "image/webp"

    return DEFAULT_IMAGE_MIME
