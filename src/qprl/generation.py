"""Batched generation with Hugging Face `generate`, shared by eval, DPO
sampling and GRPO rollouts.

Prompts are left-padded (decoder-only models must continue from the last real
token). Returned completion ids are right-padded with `pad_token_id` and come
with a mask that is 1 up to and including the first EOS, 0 after.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class GenerationBatch:
    prompt_ids: torch.Tensor  # (N, P) left-padded
    prompt_mask: torch.Tensor  # (N, P)
    completion_ids: torch.Tensor  # (N, C) right-padded
    completion_mask: torch.Tensor  # (N, C) 1 for real tokens incl. first EOS
    texts: list[str]  # decoded completions (special tokens removed)
    truncated: torch.Tensor  # (N,) bool, True if no EOS within max_new_tokens


def completion_mask_from_ids(completion_ids: torch.Tensor, eos_ids: Sequence[int]) -> torch.Tensor:
    """1 for tokens up to and including the first EOS, 0 afterwards.

    If a row has no EOS, every position is kept (the row was truncated).
    """
    is_eos = torch.zeros_like(completion_ids, dtype=torch.bool)
    for e in eos_ids:
        is_eos |= completion_ids == e
    C = completion_ids.size(1)
    idx = torch.arange(C, device=completion_ids.device).expand_as(completion_ids)
    first = torch.where(is_eos, idx, torch.full_like(idx, C)).min(dim=1).values
    return (idx <= first.unsqueeze(1)).long()


def eos_token_ids(tokenizer: Any, model: Any) -> list[int]:
    ids: set[int] = set()
    gen_eos = getattr(getattr(model, "generation_config", None), "eos_token_id", None)
    if isinstance(gen_eos, int):
        ids.add(gen_eos)
    elif isinstance(gen_eos, list):
        ids.update(int(i) for i in gen_eos)
    if tokenizer.eos_token_id is not None:
        ids.add(int(tokenizer.eos_token_id))
    return sorted(ids)


@torch.no_grad()
def generate(
    model: Any,
    tokenizer: Any,
    prompts: Sequence[str],
    max_new_tokens: int,
    do_sample: bool,
    temperature: float = 1.0,
    top_p: float = 1.0,
    num_return_sequences: int = 1,
    device: torch.device | None = None,
) -> GenerationBatch:
    """Generate completions. With num_return_sequences=G the output rows are
    grouped: rows [i*G, (i+1)*G) belong to prompt i (GRPO relies on this)."""
    device = device or next(model.parameters()).device
    tokenizer.padding_side = "left"
    enc = tokenizer(list(prompts), return_tensors="pt", padding=True, add_special_tokens=False)
    enc = {k: v.to(device) for k, v in enc.items()}
    eos = eos_token_ids(tokenizer, model)
    was_training = model.training
    model.eval()
    gen_kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "num_return_sequences": num_return_sequences,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": eos,
        "use_cache": True,
    }
    if do_sample:
        gen_kwargs.update(temperature=temperature, top_p=top_p, top_k=0)
    out = model.generate(**enc, **gen_kwargs)
    if was_training:
        model.train()
    P = enc["input_ids"].size(1)
    prompt_ids = enc["input_ids"].repeat_interleave(num_return_sequences, dim=0)
    prompt_mask = enc["attention_mask"].repeat_interleave(num_return_sequences, dim=0)
    completion_ids = out[:, P:]
    if completion_ids.size(1) == 0:  # degenerate: nothing generated
        completion_ids = torch.full(
            (out.size(0), 1), tokenizer.pad_token_id, dtype=out.dtype, device=out.device
        )
    completion_mask = completion_mask_from_ids(completion_ids, eos)
    has_eos = torch.zeros(completion_ids.size(0), dtype=torch.bool, device=completion_ids.device)
    for e in eos:
        has_eos |= (completion_ids == e).any(dim=1)
    # Pad everything after the first EOS so stale tokens never leak into training.
    completion_ids = completion_ids.masked_fill(completion_mask == 0, tokenizer.pad_token_id)
    texts = tokenizer.batch_decode(completion_ids, skip_special_tokens=True)
    return GenerationBatch(
        prompt_ids=prompt_ids,
        prompt_mask=prompt_mask,
        completion_ids=completion_ids,
        completion_mask=completion_mask,
        texts=texts,
        truncated=~has_eos,
    )


def generate_in_batches(
    model: Any,
    tokenizer: Any,
    prompts: Sequence[str],
    batch_size: int,
    max_new_tokens: int,
    do_sample: bool = False,
    temperature: float = 1.0,
    top_p: float = 1.0,
    num_return_sequences: int = 1,
) -> list[str]:
    """Text-only convenience wrapper for eval and DPO sampling."""
    texts: list[str] = []
    for i in range(0, len(prompts), batch_size):
        batch = generate(
            model,
            tokenizer,
            prompts[i : i + batch_size],
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            num_return_sequences=num_return_sequences,
        )
        texts.extend(batch.texts)
    return texts
