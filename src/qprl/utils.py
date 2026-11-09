"""Small shared helpers: JSONL IO, seeding, config loading, model loading."""

from __future__ import annotations

import json
import os
import random
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> int:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def seed_everything(seed: int) -> None:
    """Seed python, numpy and torch (CPU + CUDA)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def load_config(path: str | Path, overrides: list[str] | None = None) -> DictConfig:
    """Load a YAML config and apply `key=value` dotlist overrides from the CLI.

    A config may name a `base:` file (relative path) whose values it overrides.
    """
    path = Path(path)
    cfg = OmegaConf.load(path)
    assert isinstance(cfg, DictConfig)
    if "base" in cfg:
        base = load_config(path.parent / str(cfg.base))
        cfg = OmegaConf.merge(base, cfg)
        assert isinstance(cfg, DictConfig)
        del cfg["base"]
    if overrides:
        merged = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
        assert isinstance(merged, DictConfig)
        cfg = merged
    return cfg


def resolve_dtype(name: str) -> torch.dtype:
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[name]


def pick_device(requested: str = "auto") -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model_and_tokenizer(
    name_or_path: str, dtype: str = "fp32", gradient_checkpointing: bool = False
) -> tuple[Any, Any]:
    """Load a causal LM and its tokenizer for full-parameter training.

    Weights are loaded in `dtype`. For mixed precision (the default) load fp32
    master weights and let autocast run the matmuls in bf16; for the larger
    optional configs load bf16 weights directly ("bf16-true").
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(name_or_path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(name_or_path, dtype=resolve_dtype(dtype))
    model.config.use_cache = False
    if gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    return model, tok
