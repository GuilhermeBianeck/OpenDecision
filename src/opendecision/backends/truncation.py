"""Whether a backend may trim a state that does not fit its context.

By default the backends trim only the state, keep the question and options
whole, and warn. A caller that must not lose context silently, such as the
shared decision route, wraps its call in :func:`refuse_truncation`; a backend
that would have trimmed then raises :class:`ContextOverflow` instead. The flag
is per thread, because inference runs in the calling thread, so one request
refusing truncation never changes another's behaviour.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager

from opendecision.errors import ContextOverflow

_local = threading.local()


@contextmanager
def refuse_truncation() -> Iterator[None]:
    previous = getattr(_local, "refuse", False)
    _local.refuse = True
    try:
        yield
    finally:
        _local.refuse = previous


def check_truncation(tokens: int, limit: int) -> None:
    """Call where a backend is about to trim; raises if trimming is refused."""
    if getattr(_local, "refuse", False):
        raise ContextOverflow(tokens, limit)
