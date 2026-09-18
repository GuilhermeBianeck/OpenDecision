"""Pinned ModernBERT natural language inference baseline."""

from typing import Any

from opendecision.backends.catalog import backend_options
from opendecision.backends.transformers import TransformersBackend


class ModernBertBackend(TransformersBackend):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**backend_options("base"), **kwargs)
