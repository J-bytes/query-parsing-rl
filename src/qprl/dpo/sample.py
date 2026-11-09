"""Stage 2a: build DPO preference pairs from the SFT model's own samples.

For each query in the DPO pool: sample k completions from the SFT policy
(temperature > 0), score each with the verifiable reward (field F1 + format
bonus), and keep (best, worst) as (chosen, rejected) if their reward gap is at
least `min_margin`. Queries where all k samples tie are skipped: they carry no
preference signal.

Why on-policy pairs rather than (gold, model output)? DPO's implicit reward is
log pi/pi_ref on the chosen and rejected completions. Pairs sampled from pi_ref
itself have reasonable likelihood under it, so the log-ratios are well-scaled.
Using the gold JSON as "chosen" is closer to SFT and is offered as an ablation
(`gold_chosen=true`).

Output JSONL rows: {id, query, prompt, chosen, rejected, chosen_reward,
rejected_reward, k_rewards}.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from omegaconf import DictConfig

from qprl.generation import generate_in_batches
from qprl.tasks import make_task
from qprl.utils import (
    load_config,
    load_model_and_tokenizer,
    pick_device,
    read_jsonl,
    seed_everything,
    write_jsonl,
)


def select_pair(
    completions: list[str], rewards: list[float], min_margin: float
) -> tuple[int, int] | None:
    """Indices (chosen, rejected): highest and lowest reward. Ties broken by
    first occurrence, so the result is deterministic. None if the gap is small."""
    hi = max(range(len(rewards)), key=lambda i: (rewards[i], -i))
    lo = min(range(len(rewards)), key=lambda i: (rewards[i], i))
    if rewards[hi] - rewards[lo] < min_margin or completions[hi] == completions[lo]:
        return None
    return hi, lo


def build_pairs(cfg: DictConfig) -> dict[str, Any]:
    seed_everything(int(cfg.seed))
    task = make_task(str(cfg.task), **dict(cfg.get("reward", {})))
    model, tok = load_model_and_tokenizer(str(cfg.sft_model), str(cfg.sample.get("dtype", "bf16")))
    model.to(pick_device())
    rows = read_jsonl(str(cfg.data.pool))
    if cfg.data.get("max_pool"):
        rows = rows[: int(cfg.data.max_pool)]
    k = int(cfg.sample.k)
    prompts = [task.prompt(tok, r) for r in rows]
    texts = generate_in_batches(
        model, tok, prompts,
        batch_size=int(cfg.sample.batch_size),
        max_new_tokens=int(cfg.max_completion_tokens),
        do_sample=True,
        temperature=float(cfg.sample.temperature),
        top_p=float(cfg.sample.top_p),
        num_return_sequences=k,
    )  # fmt: skip
    pairs: list[dict[str, Any]] = []
    stats: dict[str, Any] = {"queries": len(rows), "all_tied": 0, "pairs": 0, "gold_chosen": 0}
    all_rewards: list[float] = []
    for i, row in enumerate(rows):
        comps = texts[i * k : (i + 1) * k]
        rewards = [task.reward(c, row)["reward"] for c in comps]
        all_rewards.extend(rewards)
        best_possible = task.reward(task.target(row), row)["reward"]
        choice = select_pair(comps, rewards, float(cfg.sample.min_margin))
        if bool(cfg.sample.get("gold_chosen", False)) and max(rewards) < best_possible:
            lo = min(range(k), key=lambda j: (rewards[j], j))
            chosen, chosen_r, rejected, rejected_r = (
                task.target(row),
                best_possible,
                comps[lo],
                rewards[lo],
            )
            stats["gold_chosen"] += 1
        elif choice is None:
            stats["all_tied"] += 1
            continue
        else:
            hi, lo = choice
            chosen, chosen_r, rejected, rejected_r = comps[hi], rewards[hi], comps[lo], rewards[lo]
        pairs.append({
            "id": row["id"], "query": row.get("query"), "prompt": prompts[i],
            "chosen": chosen, "rejected": rejected,
            "chosen_reward": chosen_r, "rejected_reward": rejected_r, "k_rewards": rewards,
        })  # fmt: skip
    stats["pairs"] = len(pairs)
    stats["mean_sample_reward"] = sum(all_rewards) / max(len(all_rewards), 1)
    out = Path(str(cfg.data.pairs))
    write_jsonl(out, pairs)
    out.with_suffix(".stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--config", default="configs/dpo.yaml")
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()
    build_pairs(load_config(args.config, args.overrides))


if __name__ == "__main__":
    main()
