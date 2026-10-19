"""Dependency-free HTTP client for a running ``opendecision serve``.

The in-process :class:`~opendecision.DecisionModel` needs no server; this
client is for programs that share one resident model over loopback, or that
already speak the shared ``/v1/systemone`` format to another runtime.
"""

from __future__ import annotations

import gzip
import json
import urllib.error
import urllib.request
from typing import Any

DEFAULT_BASE_URL = "http://127.0.0.1:8042"
# Bodies above this size are gzip-compressed; the server accepts one layer.
COMPRESS_ABOVE_BYTES = 65_536


class OpenDecisionHTTPError(RuntimeError):
    """The server answered with an error status; ``body`` keeps its detail."""

    def __init__(self, status: int, body: Any) -> None:
        super().__init__(f"OpenDecision HTTP request failed ({status}): {body}")
        self.status = status
        self.body = body


class Client:
    """Typed-JSON calls to a local OpenDecision server. No API key is needed."""

    def __init__(self, base_url: str = DEFAULT_BASE_URL, *, timeout: float = 30.0) -> None:
        if not base_url.startswith(("http://", "https://")):
            raise ValueError("base_url must be an HTTP or HTTPS URL")
        if not timeout > 0:
            raise ValueError("timeout must be positive")
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _request(self, path: str, body: Any = None) -> Any:
        headers = {"Accept": "application/json"}
        data = None
        if body is not None:
            data = json.dumps(body, allow_nan=False).encode()
            headers["Content-Type"] = "application/json"
            if len(data) > COMPRESS_ABOVE_BYTES:
                data = gzip.compress(data)
                headers["Content-Encoding"] = "gzip"
        request = urllib.request.Request(
            self.base_url + path,
            data=data,
            headers=headers,
            method="GET" if data is None else "POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read() or b"null")
        except urllib.error.HTTPError as error:
            text = error.read().decode("utf-8", "replace")
            try:
                detail: Any = json.loads(text)
            except ValueError:
                detail = text
            raise OpenDecisionHTTPError(error.code, detail) from None

    def health(self) -> dict[str, Any]:
        return self._request("/health")

    def systemone(self, request: dict[str, Any]) -> dict[str, Any]:
        """Send a request in the shared format and return the response JSON.

        ``request`` is ``{"state": ..., "model": ..., "questions": {...}}``;
        answers come back under the same question keys, with ``usage``.
        """
        return self._request("/v1/systemone", request)

    def ask(self, *, state: Any, questions: dict[str, Any]) -> dict[str, Any]:
        """The native multi-question call, with abstention and raw scores."""
        return self._request("/v1/ask", {"state": state, "questions": questions})
