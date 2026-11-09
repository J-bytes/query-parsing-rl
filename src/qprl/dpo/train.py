"""Stage 2b: DPO training (full parameters) in PyTorch Lightning.

The reference policy is the SFT model, which is also the policy's
initialisation. So before training we run one no-grad pass over all pairs with
the initial weights and store log pi_ref(chosen) and log pi_ref(rejected). The
reference model then never needs to be in GPU memory during training: half the
weights, and no second forward per step.

Each batch stacks chosen and rejected rows: [chosen_1..chosen_B, rejected_1..rejected_B],
laid out as [left-padded prompt | right-padded completion].
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import lightning as L
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset

from qprl.dpo.loss import dpo_loss
from qprl.logprobs import completion_token_logprobs, masked_sum
from qprl.tasks import make_task
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
from qprl.utils import (
    load_config,
    load_model_and_tokenizer,
    pick_device,
    read_jsonl,
    seed_everything,
)


class PairDataset(Dataset[dict[str, Any]]):
    def __init__(self, pairs: list[dict[str, Any]], tokenizer: Any, max_completion: int) -> None:
        self.items: list[dict[str, Any]] = []
        for p in pairs:
            pc, c = encode_prompt_completion(tokenizer, p["prompt"], p["chosen"], max_completion)
            pr, r = encode_prompt_completion(tokenizer, p["prompt"], p["rejected"], max_completion)
            # Use the shorter shared prefix as the prompt, like TRL, in case the two
            # tokenisations split the boundary differently.
            k = min(len(pc), len(pr))
            self.items.append({
                "prompt": pc[:k], "chosen": pc[k:] + c, "rejected": pr[k:] + r,
                "ref_chosen": 0.0, "ref_rejected": 0.0,
            })  # fmt: skip

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> dict[str, Any]:
        return self.items[i]


def dpo_collate(batch: list[dict[str, Any]], pad_id: int) -> dict[str, torch.Tensor]:
    prompts = [b["prompt"] for b in batch] * 2
    comps = [b["chosen"] for b in batch] + [b["rejected"] for b in batch]
    out = collate_prompt_completion(prompts, comps, pad_id)
    out["ref_chosen"] = torch.tensor([b["ref_chosen"] for b in batch], dtype=torch.float32)
    out["ref_rejected"] = torch.tensor([b["ref_rejected"] for b in batch], dtype=torch.float32)
    return out


def sequence_logps(model: Any, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    """Summed completion log-prob per row: (2B,)."""
    mask = batch["completion_mask"]
    per_tok = completion_token_logprobs(
        model, batch["input_ids"], batch["attention_mask"], mask.size(1)
    )
    return masked_sum(per_tok, mask.to(per_tok.dtype))


@torch.no_grad()
def precompute_reference(model: Any, ds: PairDataset, pad_id: int, batch_size: int) -> None:
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    for i in range(0, len(ds), batch_size):
        items = ds.items[i : i + batch_size]
        batch = {k: v.to(device) for k, v in dpo_collate(items, pad_id).items()}
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logps = sequence_logps(model, batch).float().cpu()
        B = len(items)
        for j, it in enumerate(items):
            it["ref_chosen"] = float(logps[j])
            it["ref_rejected"] = float(logps[B + j])
    model.train(was_training)


class DPOModule(L.LightningModule):
    def __init__(self, policy: Any, cfg: DictConfig, total_steps: int) -> None:
        super().__init__()
        self.policy = policy
        self.cfg = cfg
        self.total_steps = total_steps

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        logps = sequence_logps(self.policy, batch)
        B = batch["ref_chosen"].size(0)
        out = dpo_loss(
            logps[:B], logps[B:], batch["ref_chosen"], batch["ref_rejected"],
            beta=float(self.cfg.beta), label_smoothing=float(self.cfg.get("label_smoothing", 0.0)),
        )  # fmt: skip
        self.log("train/loss", out.loss, prog_bar=True)
        self.log("train/reward_accuracy", out.reward_accuracy, prog_bar=True)
        self.log("train/reward_margin", out.reward_margin)
        self.log("train/chosen_reward", out.chosen_rewards.mean())
        self.log("train/rejected_reward", out.rejected_rewards.mean())
        self.log("train/chosen_logps", logps[:B].detach().mean())
        self.log("train/rejected_logps", logps[B:].detach().mean())
        return out.loss

    def configure_optimizers(self) -> Any:
        opt = build_optimizer(self.policy.parameters(), self.cfg.optim)
        sched = build_scheduler(opt, self.cfg.optim, self.total_steps)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "step"}}


def train(cfg: DictConfig) -> Path:
    seed_everything(int(cfg.seed))
    dump_config(cfg)
    task = make_task(str(cfg.task), **dict(cfg.get("reward", {})))
    policy, tok = load_model_and_tokenizer(
        str(cfg.sft_model), str(cfg.get("dtype", "fp32")), bool(cfg.gradient_checkpointing)
    )
    pairs = read_jsonl(str(cfg.data.pairs))
    ds = PairDataset(pairs, tok, int(cfg.max_completion_tokens))
    device = pick_device()
    policy.to(device)
    precompute_reference(policy, ds, tok.pad_token_id, int(cfg.get("ref_batch_size", 16)))
    dev_rows = read_jsonl(str(cfg.data.dev))[: int(cfg.eval.num_dev)]
    g = torch.Generator().manual_seed(int(cfg.seed))
    dl = DataLoader(
        ds, batch_size=int(cfg.batch_size), shuffle=True, generator=g,
        collate_fn=lambda b: dpo_collate(b, tok.pad_token_id),
    )  # fmt: skip
    accum = int(cfg.grad_accum)
    total = int(cfg.get("max_steps") or max(len(dl) // accum, 1) * int(cfg.epochs))
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
        accelerator="auto",
        devices=1,
    )
    # from_pretrained returns eval mode; gradient checkpointing only runs in train mode.
    policy.train()
    trainer.fit(DPOModule(policy, cfg, total), train_dataloaders=dl)
    save_hf(policy, tok, out / "last", {"step": trainer.global_step})
    return out / "best"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--config", default="configs/dpo.yaml")
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()
    train(load_config(args.config, args.overrides))


if __name__ == "__main__":
    main()
