"""Request bodies: bounded, decompressed, and for strict routes checked as JSON.

Every route gets a byte cap that also applies after decompression, so a small
compressed body cannot expand past it. Routes listed as strict additionally
get the status codes the shared decision format documents:

* 400 — the body is not JSON, or its compression is corrupt;
* 413 — the body, before or after decompression, is over the cap;
* 415 — the body is not ``application/json``, or uses an unknown or stacked
  ``Content-Encoding``;
* 422 — the JSON is well formed but nested too deeply or holds a number that
  is not finite (validation errors also answer 422, from the route itself).
"""

from __future__ import annotations

import io
import json
import math
import zlib
from typing import Any

from fastapi.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

MAX_BODY_BYTES = 1_048_576
MAX_JSON_DEPTH = 64


class BodyError(Exception):
    def __init__(self, status: int, detail: str) -> None:
        super().__init__(detail)
        self.status = status
        self.detail = detail


def _size(n: int) -> str:
    return f"{n // 1_048_576} MiB" if n % 1_048_576 == 0 else f"{n} bytes"


def _zstd_reader(body: bytes, max_bytes: int) -> bytes:
    try:  # Python 3.14+
        from compression import zstd

        decompressor = zstd.ZstdDecompressor()
        out = decompressor.decompress(body, max_length=max_bytes + 1)
        if len(out) <= max_bytes and not decompressor.eof:
            raise BodyError(400, "Request body is a truncated zstd stream")
        return out
    except ImportError:
        pass
    try:
        import zstandard
    except ImportError:
        raise BodyError(415, "zstd request bodies need Python 3.14 or the zstandard package")
    reader = zstandard.ZstdDecompressor().stream_reader(io.BytesIO(body))
    return reader.read(max_bytes + 1)


def decode_body(body: bytes, encoding: str, max_bytes: int) -> bytes:
    """Undo one ``Content-Encoding`` layer without inflating past ``max_bytes``."""
    coding = encoding.strip().lower()
    if coding in ("", "identity"):
        return body
    if "," in coding:
        raise BodyError(415, "Only one Content-Encoding layer is supported")
    try:
        if coding in ("gzip", "x-gzip"):
            decompressor = zlib.decompressobj(wbits=16 + zlib.MAX_WBITS)
            out = decompressor.decompress(body, max_bytes + 1)
            if len(out) <= max_bytes and not decompressor.eof:
                raise BodyError(400, "Request body is a truncated gzip stream")
        elif coding == "zstd":
            out = _zstd_reader(body, max_bytes)
        else:
            raise BodyError(415, f"Unsupported Content-Encoding {coding!r}")
    except BodyError:
        raise
    except Exception as error:  # zlib.error, zstd errors
        raise BodyError(400, f"Request body is not valid {coding}: {error}") from error
    if len(out) > max_bytes:
        raise BodyError(413, f"Decompressed request body exceeds {_size(max_bytes)}")
    return out


def _reject_constant(name: str) -> Any:
    raise ValueError(f"{name} is not valid JSON")


def check_json(body: bytes, content_type: str) -> None:
    """Raise :class:`BodyError` unless ``body`` is acceptable strict JSON."""
    media = content_type.split(";", 1)[0].strip().lower()
    if media != "application/json" and not media.endswith("+json"):
        raise BodyError(415, "Request body must be application/json")
    try:
        value = json.loads(body, parse_constant=_reject_constant)
    except (ValueError, UnicodeDecodeError) as error:
        raise BodyError(400, f"Request body is not valid JSON: {error}") from error
    # Iterative walk, so a hostile document cannot exhaust the Python stack.
    stack = [(value, 1)]
    while stack:
        item, depth = stack.pop()
        if isinstance(item, float) and not math.isfinite(item):
            raise BodyError(422, "Numbers must be finite")
        if isinstance(item, (dict, list)):
            if depth > MAX_JSON_DEPTH:
                raise BodyError(422, f"JSON nesting exceeds {MAX_JSON_DEPTH} levels")
            children = item.values() if isinstance(item, dict) else item
            stack.extend((child, depth + 1) for child in children)


class RequestBody:
    """Bound and decode request bodies, including ones without a Content-Length."""

    def __init__(
        self,
        app: ASGIApp,
        max_bytes: int = MAX_BODY_BYTES,
        path_limits: dict[str, int] | None = None,
        strict_paths: frozenset[str] = frozenset(),
    ) -> None:
        self.app = app
        self.max_bytes = max_bytes
        self.path_limits = path_limits or {}
        self.strict_paths = strict_paths

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        path = scope.get("path", "")
        limit = self.path_limits.get(path, self.max_bytes)
        headers = dict(scope.get("headers", []))

        async def refuse(status: int, detail: str) -> None:
            await JSONResponse({"detail": detail}, status_code=status)(scope, receive, send)

        try:
            length = int(headers.get(b"content-length", b"0"))
        except ValueError:
            await refuse(400, "Invalid Content-Length")
            return
        if length < 0 or length > limit:
            await refuse(413, f"Request body exceeds {_size(limit)}")
            return
        raw = bytearray()
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            raw.extend(message.get("body", b""))
            if len(raw) > limit:
                await refuse(413, f"Request body exceeds {_size(limit)}")
                return
            if not message.get("more_body", False):
                break

        encoding = headers.get(b"content-encoding", b"").decode("latin-1")
        try:
            body = decode_body(bytes(raw), encoding, limit)
            if path in self.strict_paths and scope.get("method") == "POST":
                check_json(body, headers.get(b"content-type", b"").decode("latin-1"))
        except BodyError as error:
            await refuse(error.status, error.detail)
            return
        if encoding.strip():
            # Downstream sees the plain body, as if it had been sent uncompressed.
            scope = dict(scope)
            scope["headers"] = [
                (name, value)
                for name, value in scope["headers"]
                if name not in (b"content-encoding", b"content-length")
            ] + [(b"content-length", str(len(body)).encode())]

        consumed = False

        async def bounded_receive() -> dict[str, Any]:
            nonlocal consumed
            if consumed:
                return await receive()
            consumed = True
            return {"type": "http.request", "body": body, "more_body": False}

        await self.app(scope, bounded_receive, send)
