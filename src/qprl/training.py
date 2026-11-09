"""Lightning plumbing shared by the SFT, DPO and GRPO stages.

* build_optimizer: AdamW (fused on CUDA) or bitsandbytes 8-bit AdamW.
* GenerationEvalCallback: every N steps, greedy-decode a fixed dev subset with
  the *same* harness as the final eval and log its metrics. This is the curve
  that matters (dev field F1 vs step), not the training loss.
* Best-checkpoint saving in Hugging Face format, so the next stage loads it with
  `from_pretrained` and no Lightning code.
* build_loggers: CSV always, TensorBoard always, W&B if enabled.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import lightning as L
import torch
from lightning.pytorch.loggers import CSVLogger, Logger, TensorBoardLogger
from omegaconf import DictConfig, OmegaConf

from qprl.generation import generate_in_batches
from qprl.tasks import Task


def build_optimizer(params: Any, cfg: DictConfig) -> torch.optim.Optimizer:
    name = str(cfg.get("optimizer", "adamw"))
    lr, wd = float(cfg.lr), float(cfg.get("weight_decay", 0.0))
    betas = tuple(cfg.get("betas", (0.9, 0.999)))
    if name == "adamw8bit":
        import bitsandbytes as bnb

        opt: torch.optim.Optimizer = bnb.optim.AdamW8bit(
            params, lr=lr, weight_decay=wd, betas=betas
        )
        return opt
    return torch.optim.AdamW(
        params, lr=lr, weight_decay=wd, betas=betas, fused=torch.cuda.is_available()
    )


def warmup_cosine(step: int, warmup: int, total: int, min_ratio: float = 0.1) -> float:
    """LR multiplier: linear warmup to 1, cosine decay to min_ratio."""
    if step < warmup:
        return (step + 1) / max(warmup, 1)
    progress = min(1.0, (step - warmup) / max(total - warmup, 1))
    return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress))


def build_scheduler(
    opt: torch.optim.Optimizer, cfg: DictConfig, total_steps: int
) -> torch.optim.lr_scheduler.LambdaLR:
    kind = str(cfg.get("lr_schedule", "cosine"))
    warmup = int(cfg.get("warmup_steps", 0))
    if kind == "constant":
        return torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / max(warmup, 1)))
    min_ratio = float(cfg.get("min_lr_ratio", 0.1))
    return torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: warmup_cosine(s, warmup, total_steps, min_ratio)
    )


def build_loggers(cfg: DictConfig) -> list[Logger]:
    out = Path(str(cfg.output_dir))
    loggers: list[Logger] = [
        CSVLogger(save_dir=str(out), name="csv", version=""),
        TensorBoardLogger(save_dir=str(out), name="tb", version=""),
    ]
    if cfg.get("wandb", False):
        from lightning.pytorch.loggers import WandbLogger

        loggers.append(WandbLogger(project="query-parsing-rl", name=str(cfg.get("run_name", ""))))
    return loggers


def precision_for(cfg: DictConfig) -> str:
    p = str(cfg.get("precision", "bf16-mixed"))
    if not torch.cuda.is_available() and p.startswith("bf16"):
        return "32-true"  # CPU smoke tests
    return p


def save_hf(model: Any, tokenizer: Any, path: Path, extra: dict[str, Any] | None = None) -> None:
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path, safe_serialization=True)
    tokenizer.save_pretrained(path)
    if extra:
        (path / "qprl_meta.json").write_text(json.dumps(extra, indent=2, default=str))


def dump_config(cfg: DictConfig) -> None:
    out = Path(str(cfg.output_dir))
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.yaml").write_text(OmegaConf.to_yaml(cfg))


class GenerationEvalCallback(L.Callback):
    """Greedy-decode `rows` every `every_n_steps` optimizer steps and log metrics
    under `dev/`. Saves an HF checkpoint whenever `monitor` improves."""

    def __init__(
        self,
        task: Task,
        rows: Sequence[dict[str, Any]],
        tokenizer: Any,
        every_n_steps: int,
        batch_size: int,
        max_new_tokens: int,
        save_dir: Path | None,
        monitor: str = "micro_f1",
    ) -> None:
        self.task = task
        self.rows = list(rows)
        self.tokenizer = tokenizer
        self.every = every_n_steps
        self.batch_size = batch_size
        self.max_new_tokens = max_new_tokens
        self.save_dir = save_dir
        self.monitor = monitor
        self.best = -math.inf
        self.history: list[dict[str, Any]] = []

    def run(self, trainer: L.Trainer, pl_module: L.LightningModule) -> dict[str, Any]:
        model = pl_module.policy
        prompts = [self.task.prompt(self.tokenizer, r) for r in self.rows]
        texts = generate_in_batches(
            model, self.tokenizer, prompts, self.batch_size, self.max_new_tokens
        )
        metrics = self.task.evaluate(texts, self.rows)
        flat = {f"dev/{k}": float(v) for k, v in metrics.items() if isinstance(v, (int, float))}
        step = trainer.global_step
        for logger in trainer.loggers:
            logger.log_metrics(flat, step=step)
        self.history.append({"step": step, **flat})
        score = float(metrics.get(self.monitor, -math.inf))
        if self.save_dir is not None and score > self.best:
            self.best = score
            save_hf(model, self.tokenizer, self.save_dir, {"step": step, **flat})
        print(f"[dev @ step {step}] " + " ".join(f"{k}={v:.4f}" for k, v in flat.items()))
        return metrics

    def on_train_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        self.run(trainer, pl_module)  # step-0 baseline: where the stage starts

    def on_train_batch_end(
        self, trainer: L.Trainer, pl_module: L.LightningModule, *args: Any
    ) -> None:
        step = trainer.global_step
        if (
            step > 0
            and step % self.every == 0
            and (not self.history or self.history[-1]["step"] != step)
        ):
            self.run(trainer, pl_module)

    def on_train_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        if not self.history or self.history[-1]["step"] != trainer.global_step:
            self.run(trainer, pl_module)
