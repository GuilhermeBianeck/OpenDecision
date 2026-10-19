"""Option-order averaging as a setting of the lettered decoder backends."""

from __future__ import annotations

import pytest

from opendecision import DecisionModel
from opendecision.cli import build_parser
from opendecision.errors import BackendError
from opendecision.registry import DECODERS, create_backend


def test_decoders_accept_order_averaging():
    assert {"decoder", "qwen35", "lightning"} <= DECODERS
    assert create_backend("lightning", permutations=3).permutations == 3


def test_order_invariant_backends_reject_it_clearly():
    with pytest.raises(BackendError, match="already order-invariant"):
        create_backend("base", permutations=2)


def test_averaging_disables_a_shipped_single_order_profile():
    assert DecisionModel("lightning").calibration is not None
    assert DecisionModel("lightning", permutations=2).calibration is None


def test_cli_flag_reaches_the_backend():
    from opendecision import cli

    args = build_parser().parse_args(
        ["consistency", "--model", "lightning", "--order-averaging", "2"]
    )
    assert cli._load_model(args).backend.permutations == 2
    args = build_parser().parse_args(["consistency", "--model", "base"])
    assert args.order_averaging == 1
