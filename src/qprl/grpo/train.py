"""Stage 3: GRPO training loop, PyTorch Lightning with manual optimisation.

The maths lives in `grpo/algorithm.py` (pure tensor functions). This file is
the rollout / update loop around it. One training step:

  1. Take P prompts. Sample G completions for each with the current policy
     (rows [k*G, (k+1)*G) belong to prompt k).
  2. Score every completion with the task's verifiable reward.
  3. Group-relative advantages (one scalar per completion).
  4. Without grad: log-probs under pi_old (only needed if num_iterations > 1)
     and under the frozen reference pi_ref (only needed if beta > 0).
  5. num_iterations times: forward the batch in micro-batches with grad,
     clipped surrogate + KL loss, accumulate gradients, clip, optimizer step.

Why manual optimisation: one "batch" here produces several optimizer steps
(num_iterations), and generation happens inside the step. Lightning's automatic
loop assumes one forward/backward per batch.

Memory on one 24GB GPU with Qwen2.5-0.5B: fp32 master weights + AdamW states
(~8 GB), bf16 reference copy (~1 GB), activations for one micro-batch with
gradient checkpointing. HF `generate` is the slow part; see README for vLLM.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import lightning as L
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset

from qprl.generation import generate
from qprl.grpo.algorithm import group_advantages, grpo_loss
from qprl.logprobs import completion_token_logprobs
from qprl.tasks import Task, make_task
from qprl.training import (
    GenerationEvalCallback,
    build_loggers,
    build_optimizer,
    build_scheduler,
    dump_config,
    precision_for,
    save_hf,
)
from qprl.utils import load_config, load_model_and_tokenizer, read_jsonl, seed_everything


class PromptDataset(Dataset[dict[str, Any]]):
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> dict[str, Any]:
        return self.rows[i]


def _identity(batch: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return batch


class GRPOModule(L.LightningModule):
    def __init__(
        self,
        policy: Any,
        ref: Any | None,
        tokenizer: Any,
        task: Task,
        cfg: DictConfig,
        total_steps: int,
    ) -> None:
        super().__init__()
        self.automatic_optimization = False
        self.policy = policy
        self.ref = ref
        self.tok = tokenizer
        self.task = task
        self.cfg = cfg
        self.total_steps = total_steps
        out = Path(str(cfg.output_dir))
        out.mkdir(parents=True, exist_ok=True)
        self.rollout_log = out / "rollouts.jsonl"

    # ---------------------------------------------------------------- helpers

    @torch.no_grad()
    def _logprobs_no_grad(
        self, model: Any, ids: torch.Tensor, attn: torch.Tensor, n_comp: int, mb: int
    ) -> torch.Tensor:
        was_training = model.training
        model.eval()
        chunks = [
            completion_token_logprobs(
                model, ids[i : i + mb], attn[i : i + mb], n_comp, float(self.cfg.temperature)
            )
            for i in range(0, ids.size(0), mb)
        ]
        model.train(was_training)
        return torch.cat(chunks, dim=0)

    def _log_rollouts(
        self, rows: list[dict[str, Any]], texts: list[str], rewards: list[float]
    ) -> None:
        every = int(self.cfg.get("log_rollouts_every", 0))
        if not every or self.global_step % every:
            return
        G = int(self.cfg.num_generations)
        with open(self.rollout_log, "a", encoding="utf-8") as f:
            for k, row in enumerate(rows[:2]):  # two prompts per logged step
                rec = {
                    "step": self.global_step,
                    "id": row.get("id"),
                    "completions": texts[k * G : (k + 1) * G],
                    "rewards": rewards[k * G : (k + 1) * G],
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # ------------------------------------------------------------------ step

    def training_step(self, rows: list[dict[str, Any]], batch_idx: int) -> None:
        cfg = self.cfg
        G = int(cfg.num_generations)
        mb = int(cfg.micro_batch_size)
        max_new = int(cfg.max_completion_tokens)

        # 1. Rollouts.
        prompts = [self.task.prompt(self.tok, r) for r in rows]
        gen = generate(
            self.policy, self.tok, prompts, max_new_tokens=max_new, do_sample=True,
            temperature=float(cfg.temperature), top_p=float(cfg.get("top_p", 1.0)),
            num_return_sequences=G,
        )  # fmt: skip
        rows_rep = [r for r in rows for _ in range(G)]

        # 2. Rewards.
        scored = [self.task.reward(t, r) for t, r in zip(gen.texts, rows_rep, strict=True)]
        rewards = torch.tensor([s["reward"] for s in scored], dtype=torch.float32)

        # 3. Advantages (on CPU, tiny).
        adv = group_advantages(rewards, G, scale=cfg.advantage_scale)
        advantages = adv.advantages.to(self.device)

        ids = torch.cat([gen.prompt_ids, gen.completion_ids], dim=1)
        attn = torch.cat([gen.prompt_mask, gen.completion_mask], dim=1)
        mask = gen.completion_mask.clone().float()
        if bool(cfg.get("mask_truncated_completions", False)):
            # A completion cut off at max_new_tokens never produced an answer, so
            # its reward says little about its tokens. DAPO drops them.
            mask[gen.truncated] = 0.0
        C = gen.completion_ids.size(1)
        N = ids.size(0)

        # 4. Frozen log-probs.
        mu = int(cfg.num_iterations)
        old_logp = self._logprobs_no_grad(self.policy, ids, attn, C, mb) if mu > 1 else None
        beta = float(cfg.beta)
        ref_logp = None
        if beta > 0:
            assert self.ref is not None
            ref_logp = self._logprobs_no_grad(self.ref, ids, attn, C, mb)

        # 5. Policy updates.
        opt = self.optimizers()
        sched = self.lr_schedulers()
        aggregation = str(cfg.loss_aggregation)
        num_items = float(mask.sum().item())
        stats: dict[str, list[float]] = {
            k: []
            for k in ("loss", "policy_loss", "kl", "clip_low", "clip_high", "ratio", "grad_norm")
        }
        self.policy.train()
        for _ in range(mu):
            for i in range(0, N, mb):
                sl = slice(i, i + mb)
                logp = completion_token_logprobs(
                    self.policy, ids[sl], attn[sl], C, float(cfg.temperature)
                )
                out = grpo_loss(
                    logp=logp,
                    old_logp=None if old_logp is None else old_logp[sl],
                    ref_logp=None if ref_logp is None else ref_logp[sl],
                    advantages=advantages[sl],
                    mask=mask[sl],
                    epsilon_low=float(cfg.epsilon_low),
                    epsilon_high=float(cfg.epsilon_high),
                    beta=beta,
                    aggregation=aggregation,  # type: ignore[arg-type]
                    max_completion_length=max_new,
                    num_items_in_batch=num_items,
                    kl_ratio_correction=bool(cfg.get("kl_ratio_correction", False)),
                )
                # "dapo" already divides by the token count of the whole batch, so
                # micro-batch losses just add up. The per-sequence aggregations
                # are means over the micro-batch, so weight each by its share.
                n_rows = ids[sl].size(0)
                scale = 1.0 if aggregation == "dapo" else n_rows / N
                self.manual_backward(out.loss * scale)
                stats["loss"].append(float(out.loss.detach()) * scale)
                stats["policy_loss"].append(float(out.policy_loss) * scale)
                stats["kl"].append(float(out.mean_kl))
                stats["clip_low"].append(float(out.clip_low_fraction))
                stats["clip_high"].append(float(out.clip_high_fraction))
                stats["ratio"].append(float(out.mean_ratio))
            gn = torch.nn.utils.clip_grad_norm_(
                self.policy.parameters(), float(cfg.optim.grad_clip)
            )
            stats["grad_norm"].append(float(gn))
            opt.step()
            opt.zero_grad(set_to_none=True)
        sched.step()

        # Logging.
        lengths = gen.completion_mask.sum(1).float()
        log = {
            "train/reward_mean": rewards.mean().item(),
            "train/reward_std": rewards.std().item() if N > 1 else 0.0,
            "train/zero_std_fraction": adv.zero_std_fraction,
            "train/completion_len": lengths.mean().item(),
            "train/truncated_fraction": gen.truncated.float().mean().item(),
            "train/loss": sum(stats["loss"]) / mu,
            "train/policy_loss": sum(stats["policy_loss"]) / mu,
            "train/kl": sum(stats["kl"]) / max(len(stats["kl"]), 1),
            "train/clip_low": sum(stats["clip_low"]) / max(len(stats["clip_low"]), 1),
            "train/clip_high": sum(stats["clip_high"]) / max(len(stats["clip_high"]), 1),
            "train/ratio": sum(stats["ratio"]) / max(len(stats["ratio"]), 1),
            "train/grad_norm": sum(stats["grad_norm"]) / mu,
            "train/lr": float(sched.get_last_lr()[0]),
        }
        for key in scored[0]:
            if key != "reward":
                log[f"train/{key}"] = sum(s[key] for s in scored) / N
        self.log_dict(log, prog_bar=False, batch_size=len(rows))
        self.log("reward", log["train/reward_mean"], prog_bar=True, batch_size=len(rows))
        self._log_rollouts(rows, gen.texts, rewards.tolist())

    def configure_optimizers(self) -> Any:
        opt = build_optimizer(self.policy.parameters(), self.cfg.optim)
        sched = build_scheduler(opt, self.cfg.optim, self.total_steps)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "step"}}


def train(cfg: DictConfig) -> Path:
    seed_everything(int(cfg.seed))
    dump_config(cfg)
    task = make_task(str(cfg.task), **dict(cfg.get("reward", {})))
    policy, tok = load_model_and_tokenizer(
        str(cfg.init_model), str(cfg.get("dtype", "fp32")), bool(cfg.gradient_checkpointing)
    )
    ref = None
    if float(cfg.beta) > 0:
        ref = copy.deepcopy(policy)
        if torch.cuda.is_available():
            ref = ref.to(torch.bfloat16)
        ref.requires_grad_(False)
        ref.eval()
    rows = read_jsonl(str(cfg.data.pool))
    if cfg.data.get("max_pool"):
        rows = rows[: int(cfg.data.max_pool)]
    dev_rows = read_jsonl(str(cfg.data.dev))[: int(cfg.eval.num_dev)]
    g = torch.Generator().manual_seed(int(cfg.seed))
    dl: DataLoader[dict[str, Any]] = DataLoader(
        PromptDataset(rows), batch_size=int(cfg.prompts_per_step), shuffle=True, generator=g,
        collate_fn=_identity, drop_last=True,
    )  # fmt: skip
    # Lightning counts optimizer steps; each batch makes num_iterations of them.
    batches = int(cfg.max_batches)
    total_opt_steps = batches * int(cfg.num_iterations)
    out = Path(str(cfg.output_dir))
    cb = GenerationEvalCallback(
        task, dev_rows, tok, int(cfg.eval.every_n_steps), int(cfg.eval.batch_size),
        int(cfg.max_completion_tokens), out / "best", str(cfg.eval.monitor),
    )  # fmt: skip
    trainer = L.Trainer(
        max_steps=total_opt_steps,
        max_epochs=10_000,
        precision=precision_for(cfg),
        logger=build_loggers(cfg),
        callbacks=[cb],
        enable_checkpointing=False,
        log_every_n_steps=1,
        accelerator="auto",
        devices=1,
    )
    # from_pretrained returns eval mode; gradient checkpointing only runs in train mode.
    policy.train()
    module = GRPOModule(policy, ref, tok, task, cfg, batches)
    trainer.fit(module, train_dataloaders=dl)
    save_hf(policy, tok, out / "last", {"step": trainer.global_step})
    return out / "best"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--config", default="configs/grpo.yaml")
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()
    train(load_config(args.config, args.overrides))


if __name__ == "__main__":
    main()
