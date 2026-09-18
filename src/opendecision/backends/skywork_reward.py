"""Pinned Skywork Qwen3 reward head, using the upstream chat template."""

from typing import Any

from opendecision.backends.catalog import backend_options
from opendecision.backends.transformers import TransformersBackend


class SkyworkRewardBackend(TransformersBackend):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**backend_options("smart"), **kwargs)
