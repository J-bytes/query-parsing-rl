"""Cross-check the from-scratch GRPO maths against TRL's GRPOTrainer.

GRPO sampling is random, so two trainers cannot be compared step for step the
way DPO can. Instead we check the two deterministic pieces directly, on TRL's
own code paths:

1. Advantages. Run TRL's `_generate_and_score_completions` with a deterministic
   reward function. It samples completions, scores them and returns its
   advantages. Re-score the same completions, feed the rewards to our
   `group_advantages`, and compare. Done for scale_rewards in {group, batch, none}.

2. Loss. Build one batch of inputs (prompt ids, completion ids, masks,
   advantages, perturbed pi_old and pi_ref log-probs), call TRL's
   `_compute_loss(model, inputs)` and our `grpo_loss` on log-probs from
   `completion_token_logprobs` with the same model. Compare the loss value and
   the gradient of every parameter. Done for every aggregation in
   {grpo, bnpo, dr_grpo, dapo}, with beta > 0 and asymmetric clipping, so the
   KL, clipping and normalisation paths are all exercised, both with and
   without TRL's `use_bias_correction_kl` (our `kl_ratio_correction`).

Prompts in check 2 have equal length (no left padding): TRL does not pass
position_ids, we do (derived from the attention mask, like `generate`).
`position_ids_effect` reports how much that choice moves log-probs on a
left-padded batch.

    make trl-check
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any

import torch

from qprl.grpo.algorithm import group_advantages, grpo_loss
from qprl.logprobs import completion_token_logprobs
from qprl.tiny_model import build_tiny_model
from qprl.utils import seed_everything

AGGREGATIONS = ["grpo", "bnpo", "dr_grpo", "dapo"]


def _length_reward(completions: list[Any], **_: Any) -> list[float]:
    """Deterministic, varied reward: character count mod 7."""
    return [float(len(str(c)) % 7) for c in completions]


def _make_trainer(model_path: str, tmp: str, **overrides: Any) -> Any:
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer

    tok = AutoTokenizer.from_pretrained(model_path)
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.float32)
    ds = Dataset.from_list([{"prompt": f"query number {i}"} for i in range(8)])
    cfg: dict[str, Any] = {
        "output_dir": tmp,
        "per_device_train_batch_size": 8,
        "num_generations": 4,
        "max_completion_length": 12,
        "temperature": 1.0,
        "beta": 0.05,
        "epsilon": 0.2,
        "epsilon_high": 0.28,
        "report_to": "none",
        "use_cpu": True,
        "bf16": False,
        "seed": 0,
        "save_strategy": "no",
        "logging_steps": 1,
    }
    cfg.update(overrides)
    args = GRPOConfig(**cfg)
    return GRPOTrainer(
        model=model, reward_funcs=_length_reward, args=args, train_dataset=ds, processing_class=tok
    )


def check_advantages(model_path: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for scale in ("group", "batch", "none"):
        seed_everything(0)
        with tempfile.TemporaryDirectory() as tmp:
            trainer = _make_trainer(model_path, tmp, scale_rewards=scale)
            trainer.model.train()
            inputs = [{"prompt": f"query number {i // 4}"} for i in range(8)]
            res = trainer._generate_and_score_completions(inputs)
            texts = trainer.processing_class.batch_decode(
                res["completion_ids"], skip_special_tokens=True
            )
            rewards = torch.tensor(_length_reward(texts))
            ours = group_advantages(rewards, 4, scale=scale).advantages
            out[scale] = float((ours - res["advantages"].cpu()).abs().max())
    return out


def _fake_batch(tok: Any, gen: torch.Generator) -> dict[str, torch.Tensor]:
    N, P, C = 8, 6, 10
    vocab = 1000  # sample ordinary token ids
    prompt_ids = torch.randint(10, vocab, (N, P), generator=gen)
    completion_ids = torch.randint(10, vocab, (N, C), generator=gen)
    lengths = torch.tensor([10, 7, 10, 3, 9, 10, 5, 8])
    completion_mask = (torch.arange(C).unsqueeze(0) < lengths.unsqueeze(1)).long()
    completion_ids = completion_ids.masked_fill(completion_mask == 0, tok.pad_token_id)
    return {
        "prompt_ids": prompt_ids,
        "prompt_mask": torch.ones_like(prompt_ids),
        "completion_ids": completion_ids,
        "completion_mask": completion_mask,
        "advantages": torch.randn(N, generator=gen),
    }


def check_loss(model_path: str) -> dict[str, dict[str, float]]:
    results: dict[str, dict[str, float]] = {}
    for agg, kl_corr in [(a, c) for a in AGGREGATIONS for c in (False, True)]:
        seed_everything(0)
        with tempfile.TemporaryDirectory() as tmp:
            trainer = _make_trainer(model_path, tmp, loss_type=agg, use_bias_correction_kl=kl_corr)
            model = trainer.model
            model.train()
            gen = torch.Generator().manual_seed(1)
            batch = _fake_batch(trainer.processing_class, gen)
            ids = torch.cat([batch["prompt_ids"], batch["completion_ids"]], 1)
            attn = torch.cat([batch["prompt_mask"], batch["completion_mask"]], 1)
            C = batch["completion_ids"].size(1)
            with torch.no_grad():
                base = completion_token_logprobs(model, ids, attn, C, temperature=1.0)
            # pi_old and pi_ref differ from pi_theta so ratios and KL are non-trivial
            # and some tokens are clipped.
            old = base + 0.3 * torch.randn(base.shape, generator=gen)
            ref = base + 0.5 * torch.randn(base.shape, generator=gen)
            num_items = batch["completion_mask"].sum()

            # TRL
            inputs = dict(batch, old_per_token_logps=old, ref_per_token_logps=ref,
                          num_items_in_batch=num_items)  # fmt: skip
            # Normally set by the HF training loop; one micro-batch, no accumulation.
            trainer.current_gradient_accumulation_steps = 1
            model.zero_grad()
            trl_loss = trainer._compute_loss(model, inputs)
            trl_loss.backward()
            trl_grads = {
                n: p.grad.detach().clone()
                for n, p in model.named_parameters()
                if p.grad is not None
            }

            # Ours
            model.zero_grad()
            logp = completion_token_logprobs(model, ids, attn, C, temperature=1.0)
            out = grpo_loss(
                logp, old, ref, batch["advantages"], batch["completion_mask"],
                epsilon_low=0.2, epsilon_high=0.28, beta=0.05, aggregation=agg,  # type: ignore[arg-type]
                max_completion_length=12, num_items_in_batch=float(num_items),
                kl_ratio_correction=kl_corr,
            )  # fmt: skip
            out.loss.backward()  # type: ignore[no-untyped-call]
            grad_diff = max(
                float((p.grad - trl_grads[n]).abs().max())
                for n, p in model.named_parameters()
                if p.grad is not None
            )
            grad_scale = max(float(g.abs().max()) for g in trl_grads.values())
            tl, ol = float(trl_loss.detach()), float(out.loss.detach())
            results[f"{agg}/kl_ratio_correction={kl_corr}"] = {
                "trl_loss": tl,
                "our_loss": ol,
                "loss_abs_diff": abs(tl - ol),
                "max_grad_abs_diff": grad_diff,
                "max_grad_abs": grad_scale,
                "clip_fraction": float(out.clip_low_fraction + out.clip_high_fraction),
            }
    return results


def position_ids_effect(model_path: str) -> float:
    """Max |logp difference| with vs without mask-derived position_ids on a
    left-padded batch. Non-zero means scoring without them is off-distribution
    relative to generation (HF generate derives positions from the mask)."""
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.float32).eval()
    ids = torch.randint(10, 1000, (2, 12), generator=torch.Generator().manual_seed(0))
    attn = torch.ones_like(ids)
    attn[0, :4] = 0  # row 0 left-padded by 4
    with torch.no_grad():
        ours = completion_token_logprobs(model, ids, attn, 5)
        plain = model(input_ids=ids, attention_mask=attn, use_cache=False).logits[:, -6:-1]
        plain = (
            torch.log_softmax(plain.float(), -1).gather(-1, ids[:, -5:].unsqueeze(-1)).squeeze(-1)
        )
    return float((ours - plain).abs().max())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--model", default=None)
    ap.add_argument("--out", type=Path, default=Path("results/trl_check_grpo.json"))
    args = ap.parse_args()
    model_path = args.model or str(build_tiny_model())
    res = {
        "advantage_max_abs_diff": check_advantages(model_path),
        "loss": check_loss(model_path),
        "position_ids_effect_left_padded": position_ids_effect(model_path),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
