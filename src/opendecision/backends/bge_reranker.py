"""Pinned BGE query/passage reranker adapted to candidate decisions."""

from typing import Any

from opendecision.backends.catalog import backend_options
from opendecision.backends.transformers import TransformersBackend


class BGERerankerBackend(TransformersBackend):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**backend_options("multilingual"), **kwargs)
