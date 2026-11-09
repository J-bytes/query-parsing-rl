"""Stage 1: supervised fine-tuning (full parameters) in PyTorch Lightning.

Loss is next-token cross-entropy on the completion tokens only (the JSON plus
EOS); prompt tokens are masked with -100. This matters for a task like this one
where the prompt (system prompt + query) is ~10x longer than the answer: an
unmasked loss would spend most of its gradient on memorising the system prompt.

    make sft                      # configs/sft.yaml
    make sft ARGS="lr=5e-6"       # dotlist overrides
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import lightning as L
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset

from qprl.tasks import Task, make_task
from qprl.tokenization import collate_prompt_completion, encode_prompt_completion
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


class SFTDataset(Dataset[tuple[list[int], list[int]]]):
    def __init__(self, rows: list[dict[str, Any]], task: Task, tokenizer: Any, max_completion: int):
        self.items = [
            encode_prompt_completion(
                tokenizer, task.prompt(tokenizer, r), task.target(r), max_completion
            )
            for r in rows
        ]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> tuple[list[int], list[int]]:
        return self.items[i]


def sft_collate(batch: list[tuple[list[int], list[int]]], pad_id: int) -> dict[str, torch.Tensor]:
    out = collate_prompt_completion([b[0] for b in batch], [b[1] for b in batch], pad_id)
    P = out["input_ids"].size(1) - out["completion_mask"].size(1)
    labels = out["input_ids"].clone()
    labels[:, :P] = -100  # never learn the prompt
    labels[:, P:][out["completion_mask"] == 0] = -100  # never learn padding
    out["labels"] = labels
    return out


class SFTModule(L.LightningModule):
    def __init__(self, policy: Any, cfg: DictConfig, total_steps: int) -> None:
        super().__init__()
        self.policy = policy
        self.cfg = cfg
        self.total_steps = total_steps

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        mask = batch["attention_mask"]
        position_ids = (mask.cumsum(-1) - 1).clamp(min=0)  # left padding safe
        out = self.policy(
            input_ids=batch["input_ids"],
            attention_mask=mask,
            position_ids=position_ids,
            labels=batch["labels"],
            use_cache=False,
        )
        loss: torch.Tensor = out.loss
        n_tok = (batch["labels"] != -100).sum()
        self.log("train/loss", loss, prog_bar=True)
        self.log("train/completion_tokens", n_tok.float())
        return loss

    def configure_optimizers(self) -> Any:
        opt = build_optimizer(self.policy.parameters(), self.cfg.optim)
        sched = build_scheduler(opt, self.cfg.optim, self.total_steps)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "step"}}


def train(cfg: DictConfig) -> Path:
    seed_everything(int(cfg.seed))
    dump_config(cfg)
    task = make_task(str(cfg.task), **dict(cfg.get("reward", {})))
    policy, tok = load_model_and_tokenizer(
        str(cfg.model), str(cfg.get("dtype", "fp32")), bool(cfg.gradient_checkpointing)
    )
    rows = read_jsonl(str(cfg.data.train))
    if cfg.data.get("max_train"):
        rows = rows[: int(cfg.data.max_train)]
    dev_rows = read_jsonl(str(cfg.data.dev))[: int(cfg.eval.num_dev)]
    ds = SFTDataset(rows, task, tok, int(cfg.max_completion_tokens))
    g = torch.Generator().manual_seed(int(cfg.seed))
    dl = DataLoader(
        ds,
        batch_size=int(cfg.batch_size),
        shuffle=True,
        generator=g,
        collate_fn=lambda b: sft_collate(b, tok.pad_token_id),
        num_workers=0,
    )
    accum = int(cfg.grad_accum)
    steps_per_epoch = max(len(dl) // accum, 1)
    total = int(cfg.get("max_steps") or steps_per_epoch * int(cfg.epochs))
    out = Path(str(cfg.output_dir))
    cb = GenerationEvalCallback(
        task, dev_rows, tok, int(cfg.eval.every_n_steps), int(cfg.eval.batch_size),
        int(cfg.max_completion_tokens), out / "best", str(cfg.eval.monitor),
    )  # fmt: skip
    trainer = L.Trainer(
        max_steps=total,
        accumulate_grad_batches=accum,
        gradient_clip_val=float(cfg.optim.grad_clip),
        precision=precision_for(cfg),
        logger=build_loggers(cfg),
        callbacks=[cb],
        enable_checkpointing=False,
        log_every_n_steps=int(cfg.get("log_every_n_steps", 10)),
        deterministic=False,
        accelerator="auto",
        devices=1,
    )
    # from_pretrained returns eval mode; gradient checkpointing only runs in train mode.
    policy.train()
    module = SFTModule(policy, cfg, total)
    trainer.fit(module, train_dataloaders=dl)
    save_hf(policy, tok, out / "last", {"step": trainer.global_step})
    print(f"best dev {cfg.eval.monitor}={cb.best:.4f}; checkpoints in {out}/best and {out}/last")
    return out / "best"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--config", default="configs/sft.yaml")
    ap.add_argument("overrides", nargs="*", help="dotlist overrides, e.g. optim.lr=1e-5")
    args = ap.parse_args()
    train(load_config(args.config, args.overrides))


if __name__ == "__main__":
    main()
