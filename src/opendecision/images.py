"""Images in a decision request: accepted forms, validation and placeholders.

Requests carry images as ``data:image/...;base64,`` URIs. The shared decision
format also allows them inside ``state`` or in a top-level ``images`` field as
bare base64 or image parts such as ``{"type": "image_url", "image_url":
{"url": ...}}``; :func:`extract_images` lifts every one out and leaves an
``[image N]`` placeholder where it was, so the state stays readable text.

Images are never fetched: an ``http(s)`` URL is rejected, because local
inference makes no network requests. Only PNG, JPEG, GIF and WebP bytes are
accepted, recognised by their signature rather than by what the request claims.
"""

from __future__ import annotations

import base64
import binascii
import re
from typing import Any

MAX_IMAGES = 4
MAX_IMAGE_BYTES = 20 << 20

_DATA_URI = re.compile(r"\s*data:image/[A-Za-z0-9.+-]+;base64,(.*)", re.S | re.I)
_BASE64 = re.compile(r"[A-Za-z0-9+/=_-]+")
# How base64 of a PNG, JPEG, GIF and WebP begins: the cheap test before any
# string inside a state is decoded, so ordinary text never qualifies.
_BASE64_HEADS = ("iVBORw0KGgo", "/9j/", "R0lGOD", "UklGR")
_PART_TYPES = ("image_url", "image", "input_image")


def image_type(raw: bytes) -> str | None:
    """The media type of supported image bytes, or None."""
    if raw.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if raw.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if raw.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    if raw[:4] == b"RIFF" and raw[8:12] == b"WEBP":
        return "image/webp"
    return None


def _decode(text: str) -> bytes | None:
    compact = re.sub(r"\s+", "", text)
    if not compact or not _BASE64.fullmatch(compact):
        return None
    padded = compact + "=" * (-len(compact) % 4)
    try:
        if "-" in compact or "_" in compact:
            return base64.urlsafe_b64decode(padded)
        return base64.b64decode(padded, validate=True)
    except (binascii.Error, ValueError):
        return None


def to_data_uri(raw: bytes) -> str:
    """Validate image bytes and return them as a data URI."""
    if len(raw) > MAX_IMAGE_BYTES:
        raise ValueError(f"an image of {len(raw)} bytes is over the {MAX_IMAGE_BYTES}-byte limit")
    media = image_type(raw)
    if media is None:
        raise ValueError("images must be PNG, JPEG, GIF or WebP")
    return f"data:{media};base64,{base64.b64encode(raw).decode('ascii')}"


def decode_data_uri(uri: str) -> bytes:
    """The bytes of a data URI produced by :func:`to_data_uri`."""
    match = _DATA_URI.match(uri)
    raw = _decode(match.group(1)) if match else None
    if raw is None:
        raise ValueError("an image must be a base64 data URI")
    return raw


def _from_string(text: str, explicit: bool) -> str | None:
    """A string as a data URI; None when it is ordinary text inside a state."""
    stripped = text.strip()
    if _DATA_URI.match(stripped):
        raw = _decode(_DATA_URI.match(stripped).group(1))
        if raw is None:
            raise ValueError("an image data URI does not carry valid base64")
        return to_data_uri(raw)
    if re.match(r"https?://", stripped, re.I):
        if explicit:
            raise ValueError(
                "image URLs are not fetched: local inference makes no network requests; "
                "send the image bytes as a base64 data URI"
            )
        return None
    if explicit:
        raw = _decode(stripped)
        if raw is None or image_type(raw) is None:
            raise ValueError("an image must be a data URI or base64 PNG, JPEG, GIF or WebP")
        return to_data_uri(raw)
    if len(stripped) < 64 or any(c.isspace() for c in stripped):
        return None
    if not stripped.startswith(_BASE64_HEADS):
        return None
    raw = _decode(stripped)
    return to_data_uri(raw) if raw is not None and image_type(raw) else None


def _from_part(value: dict[str, Any], explicit: bool) -> str | None:
    """An image part object as a data URI; None when the object is not one."""
    kind = value.get("type")
    if kind == "base64" and isinstance(value.get("data"), str):
        return _from_string(value["data"], True)
    if kind not in _PART_TYPES:
        return None
    inner = value.get(kind) if kind != "input_image" else value.get("image_url", value.get("image"))
    if isinstance(inner, dict):
        inner = inner.get("url", inner.get("data"))
    if not isinstance(inner, str):
        if explicit:
            raise ValueError(f"an image part of type {kind!r} has no url or data")
        return None
    return _from_string(inner, explicit)


def _explicit(value: Any) -> str:
    if isinstance(value, str):
        return _from_string(value, True)  # type: ignore[return-value]
    if isinstance(value, dict):
        found = _from_part(value, True)
        if found is not None:
            return found
        for key in ("url", "image_url", "data", "image"):
            inner = value.get(key)
            if isinstance(inner, dict):
                inner = inner.get("url", inner.get("data"))
            if isinstance(inner, str):
                return _from_string(inner, True)  # type: ignore[return-value]
    raise ValueError("an image entry must be a data URI, base64 image, or image part")


def _lift(value: Any, found: list[str]) -> Any:
    if isinstance(value, str):
        uri = _from_string(value, False)
        if uri is None:
            return value
        found.append(uri)
        return f"[image {len(found)}]"
    if isinstance(value, dict):
        uri = _from_part(value, False)
        if uri is not None:
            found.append(uri)
            return f"[image {len(found)}]"
        return {key: _lift(item, found) for key, item in value.items()}
    if isinstance(value, list):
        return [_lift(item, found) for item in value]
    return value


def extract_images(state: Any, images: Any = None) -> tuple[Any, list[str]]:
    """Return ``state`` with images replaced by ``[image N]`` and the images as data URIs.

    Images from the top-level ``images`` field are numbered first, then those
    found inside the state, in reading order.
    """
    found: list[str] = []
    if images is not None:
        for item in images if isinstance(images, list) else [images]:
            found.append(_explicit(item))
    lifted = _lift(state, found)
    if len(found) > MAX_IMAGES:
        raise ValueError(f"at most {MAX_IMAGES} images per request; got {len(found)}")
    return lifted, found


def validate_images(images: list[str]) -> list[str]:
    """Normalise a request's ``images`` list of data URIs, enforcing the limits."""
    if len(images) > MAX_IMAGES:
        raise ValueError(f"at most {MAX_IMAGES} images per request; got {len(images)}")
    return [to_data_uri(decode_data_uri(uri)) for uri in images]
