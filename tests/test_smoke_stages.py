"""Run every stage for a few steps on CPU with the tiny model (plumbing only)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from qprl.utils import load_config

pytestmark = pytest.mark.slow


def test_all_stages(tiny_model_path: str, tmp_path: Path) -> None:
    from qprl.dpo.sample import build_pairs
    from qprl.dpo.train import train as dpo_train
    from qprl.eval.run_eval import run as run_eval
    from qprl.grpo.train import train as grpo_train
    from qprl.sft.train import train as sft_train

    sft_out = tmp_path / "sft"
    sft_train(
        load_config("configs/smoke/sft.yaml", [f"output_dir={sft_out}", f"model={tiny_model_path}"])
    )
    assert (sft_out / "best" / "config.json").exists()
    assert (sft_out / "last" / "model.safetensors").exists()
    assert (sft_out / "csv" / "metrics.csv").stat().st_size > 0

    dpo_over = [f"sft_model={sft_out / 'last'}", f"output_dir={tmp_path / 'dpo'}",
                f"data.pairs={tmp_path / 'pairs.jsonl'}"]  # fmt: skip
    stats = build_pairs(load_config("configs/smoke/dpo.yaml", dpo_over))
    assert stats["pairs"] > 0
    dpo_train(load_config("configs/smoke/dpo.yaml", dpo_over))
    assert (tmp_path / "dpo" / "last" / "config.json").exists()

    grpo_out = tmp_path / "grpo"
    grpo_train(load_config("configs/smoke/grpo.yaml",
                           [f"init_model={sft_out / 'last'}", f"output_dir={grpo_out}"]))  # fmt: skip
    rows = (grpo_out / "csv" / "metrics.csv").read_text().splitlines()
    assert "train/zero_std_fraction" in rows[0]
    assert (grpo_out / "rollouts.jsonl").exists()

    m = run_eval(str(grpo_out / "last"), Path("tests/fixtures/queries.jsonl"), tmp_path / "res",
                 "query_parsing", 8, 8, "fp32", 4)  # fmt: skip
    assert m["n"] == 4
    assert json.loads((tmp_path / "res" / "queries_metrics.json").read_text())["model"]
