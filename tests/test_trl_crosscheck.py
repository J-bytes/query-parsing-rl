"""Numerical agreement with TRL. See dpo/trl_check.py and grpo/trl_check.py."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

pytestmark = [pytest.mark.slow, pytest.mark.trl]


def test_dpo_matches_trl(tiny_model_path: str, fixture_queries: Path) -> None:
    from transformers import AutoTokenizer

    from qprl.dpo.trl_check import compare, fixture_pairs

    tok = AutoTokenizer.from_pretrained(tiny_model_path)
    pairs = fixture_pairs(tok, fixture_queries, 4)
    res = compare(tiny_model_path, pairs, steps=3, lr=1e-3, beta=0.1)
    assert abs(res["ours"][0] - math.log(2)) < 1e-6
    assert res["ours"][-1] < res["ours"][0]  # it actually trained
    assert res["max_abs_diff"] < 1e-5


def test_grpo_advantages_match_trl(tiny_model_path: str) -> None:
    from qprl.grpo.trl_check import check_advantages

    for scale, diff in check_advantages(tiny_model_path).items():
        assert diff < 1e-5, scale


def test_grpo_loss_and_grads_match_trl(tiny_model_path: str) -> None:
    from qprl.grpo.trl_check import check_loss

    for name, r in check_loss(tiny_model_path).items():
        assert r["loss_abs_diff"] < 1e-6, name
        assert r["max_grad_abs_diff"] < 1e-6 + 1e-4 * r["max_grad_abs"], name
        assert r["clip_fraction"] > 0, name  # clipping path exercised
