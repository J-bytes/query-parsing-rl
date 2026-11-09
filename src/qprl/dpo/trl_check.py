"""Cross-check our DPO implementation against TRL's DPOTrainer.

Both trainers start from the same weights and train on the same pairs with the
same hyper-parameters. To remove every source of nondeterminism that is not
part of the algorithm:
  * full-batch updates (batch = all pairs), so data order cannot matter,
  * constant learning rate, no warmup, plain AdamW, no weight decay,
  * gradient clipping disabled in both,
  * fp32 on CPU, dropout is 0 in Qwen2.

If tokenisation, log-prob computation, reference handling and the loss all
agree, the per-step loss curves match to float tolerance. Step 0 must equal
log 2 in both (policy == reference).

    make trl-check                                     # tiny random model, CPU, ~30 s
    uv run --extra gpu python -m qprl.dpo.trl_check --model runs/sft/best --pairs data/processed/dpo_pairs.jsonl --n 16
"""

from __future__ import annotations

import argparse
import json
import math
import tempfile
from pathlib import Path
from typing import Any

import torch

from qprl.dpo.loss import dpo_loss
from qprl.dpo.train import PairDataset, dpo_collate, sequence_logps
from qprl.prompts import render_prompt
from qprl.schema import to_target_json
from qprl.tiny_model import build_tiny_model
from qprl.utils import load_model_and_tokenizer, read_jsonl, seed_everything


def fixture_pairs(tokenizer: Any, path: Path, n: int) -> list[dict[str, Any]]:
    """Chosen = gold JSON, rejected = gold JSON with one field removed (or a junk string)."""
    pairs = []
    for row in read_jsonl(path)[:n]:
        label = dict(row["label"])
        worse = dict(label)
        worse.pop(next(iter(worse)))
        pairs.append({
            "prompt": render_prompt(tokenizer, row["query"]),
            "chosen": to_target_json(label),
            "rejected": to_target_json(worse) if worse else "not json",
        })  # fmt: skip
    return pairs


def run_ours(
    model_path: str, pairs: list[dict[str, Any]], steps: int, lr: float, beta: float
) -> list[float]:
    seed_everything(0)
    model, tok = load_model_and_tokenizer(model_path, "fp32")
    ds = PairDataset(pairs, tok, max_completion=256)
    # Reference = initial policy, precomputed exactly like the real trainer.
    from qprl.dpo.train import precompute_reference

    precompute_reference(model, ds, tok.pad_token_id, batch_size=len(ds))
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    batch = dpo_collate(ds.items, tok.pad_token_id)
    losses = []
    for _ in range(steps):
        logps = sequence_logps(model, batch)
        B = len(ds)
        out = dpo_loss(logps[:B], logps[B:], batch["ref_chosen"], batch["ref_rejected"], beta)
        losses.append(float(out.loss.detach()))
        opt.zero_grad()
        out.loss.backward()  # type: ignore[no-untyped-call]
        opt.step()
    return losses


def run_trl(
    model_path: str, pairs: list[dict[str, Any]], steps: int, lr: float, beta: float
) -> list[float]:
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOConfig, DPOTrainer

    seed_everything(0)
    tok = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.float32)
    ds = Dataset.from_list([{k: p[k] for k in ("prompt", "chosen", "rejected")} for p in pairs])
    with tempfile.TemporaryDirectory() as tmp:
        args = DPOConfig(
            output_dir=tmp,
            per_device_train_batch_size=len(pairs),
            gradient_accumulation_steps=1,
            learning_rate=lr,
            lr_scheduler_type="constant",
            warmup_steps=0,
            weight_decay=0.0,
            max_grad_norm=1e9,
            optim="adamw_torch",
            max_steps=steps,
            logging_steps=1,
            beta=beta,
            loss_type=["sigmoid"],
            max_length=None,
            report_to="none",
            use_cpu=True,
            bf16=False,
            seed=0,
            save_strategy="no",
            disable_dropout=True,
        )
        trainer = DPOTrainer(model=model, args=args, train_dataset=ds, processing_class=tok)
        trainer.train()
        return [float(h["loss"]) for h in trainer.state.log_history if "loss" in h][:steps]


def compare(
    model_path: str, pairs: list[dict[str, Any]], steps: int, lr: float, beta: float
) -> dict[str, Any]:
    ours = run_ours(model_path, pairs, steps, lr, beta)
    trl = run_trl(model_path, pairs, steps, lr, beta)
    diffs = [abs(a - b) for a, b in zip(ours, trl, strict=True)]
    return {
        "ours": ours,
        "trl": trl,
        "max_abs_diff": max(diffs),
        "step0_minus_log2": ours[0] - math.log(2),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--model", default=None, help="default: tiny random Qwen2")
    ap.add_argument("--pairs", type=Path, default=None, help="JSONL with prompt/chosen/rejected")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--out", type=Path, default=Path("results/trl_check_dpo.json"))
    args = ap.parse_args()
    model_path = args.model or str(build_tiny_model())
    if args.pairs:
        pairs = read_jsonl(args.pairs)[: args.n]
    else:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(model_path)
        pairs = fixture_pairs(tok, Path("tests/fixtures/queries.jsonl"), args.n)
    res = compare(model_path, pairs, args.steps, args.lr, args.beta)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=2))
    for i, (a, b) in enumerate(zip(res["ours"], res["trl"], strict=True)):
        print(f"step {i}: ours={a:.6f}  trl={b:.6f}  |diff|={abs(a - b):.2e}")
    print(f"max |diff| = {res['max_abs_diff']:.2e}; step0 - log2 = {res['step0_minus_log2']:.2e}")


if __name__ == "__main__":
    main()
