"""Prompt/completion tokenisation and padding, shared by SFT, DPO and GRPO.

Tokenising the prompt and the completion separately can produce different ids
than tokenising their concatenation (BPE merges across the boundary). We do
what TRL does: tokenise `prompt` and `prompt + completion` and split at their
longest common prefix. The completion always ends with the EOS token text, so
the model learns to stop.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch


def common_prefix_length(a: Sequence[int], b: Sequence[int]) -> int:
    n = 0
    for x, y in zip(a, b, strict=False):
        if x != y:
            break
        n += 1
    return n


def encode_prompt_completion(
    tokenizer: Any, prompt: str, completion: str, max_completion_tokens: int | None = None
) -> tuple[list[int], list[int]]:
    """Return (prompt_ids, completion_ids). Completion includes EOS."""
    eos = tokenizer.eos_token
    if not completion.endswith(eos):
        completion = completion + eos
    p = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    pc = tokenizer(prompt + completion, add_special_tokens=False)["input_ids"]
    k = common_prefix_length(p, pc)
    prompt_ids, completion_ids = list(pc[:k]), list(pc[k:])
    if max_completion_tokens is not None:
        completion_ids = completion_ids[:max_completion_tokens]
    return prompt_ids, completion_ids


def pad_left(seqs: Sequence[Sequence[int]], pad_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    width = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), width), pad_id, dtype=torch.long)
    mask = torch.zeros((len(seqs), width), dtype=torch.long)
    for i, s in enumerate(seqs):
        if s:
            ids[i, width - len(s) :] = torch.tensor(s, dtype=torch.long)
            mask[i, width - len(s) :] = 1
    return ids, mask


def pad_right(seqs: Sequence[Sequence[int]], pad_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    width = max(1, *(len(s) for s in seqs))
    ids = torch.full((len(seqs), width), pad_id, dtype=torch.long)
    mask = torch.zeros((len(seqs), width), dtype=torch.long)
    for i, s in enumerate(seqs):
        if s:
            ids[i, : len(s)] = torch.tensor(s, dtype=torch.long)
            mask[i, : len(s)] = 1
    return ids, mask


def collate_prompt_completion(
    prompts: Sequence[Sequence[int]], completions: Sequence[Sequence[int]], pad_id: int
) -> dict[str, torch.Tensor]:
    """[left-padded prompt | right-padded completion] layout.

    Every row's completion starts at the same column, so per-token log-probs of
    the completion are simply the last C columns (see logprobs.py).
    """
    p_ids, p_mask = pad_left(prompts, pad_id)
    c_ids, c_mask = pad_right(completions, pad_id)
    return {
        "input_ids": torch.cat([p_ids, c_ids], dim=1),
        "attention_mask": torch.cat([p_mask, c_mask], dim=1),
        "completion_mask": c_mask,
    }
