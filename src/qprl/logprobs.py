"""Per-token log-probabilities of completion tokens under a causal LM.

Shared by DPO (summed per sequence) and GRPO (kept per token). Written once so
both algorithms score sequences identically.

Shapes: B = batch, L = full sequence length (prompt + completion), T = number of
trailing positions we score (the completion region, right-aligned).
"""

from __future__ import annotations

from typing import Any

import torch


def gather_logprobs(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """log softmax(logits)[targets], computed without materialising a full
    (B, T, V) log-softmax tensor in fp32 twice.

    logits: (B, T, V), targets: (B, T) -> (B, T)
    """
    logits = logits.float()
    picked = torch.gather(logits, dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
    return picked - torch.logsumexp(logits, dim=-1)


def completion_token_logprobs(
    model: Any,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    num_completion_tokens: int,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Log-probs of the last `num_completion_tokens` tokens of each row.

    The token at position t is predicted by the logits at position t-1, so we
    take logits[:, -T-1:-1] and targets input_ids[:, -T:]. Rows must be laid out
    as [left-padded prompt | right-padded completion] so that the completion
    region is the last T columns for every row. Positions that are padding are
    still scored here; callers multiply by a completion mask.

    `temperature` divides the logits, matching how sampling was done. GRPO uses
    the sampling temperature so ratios compare the same distributions (TRL does
    the same); DPO uses 1.0.

    Returns (B, T) float32.
    """
    T = num_completion_tokens
    # Positions derived from the attention mask, the same rule HF `generate`
    # uses, so scoring sees exactly the positions the sampler saw. (On the tiny
    # test model, omitting them changes log-probs by ~2e-6: see grpo/trl_check.)
    position_ids = (attention_mask.long().cumsum(-1) - 1).clamp(min=0)
    out = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        use_cache=False,
        logits_to_keep=T + 1,
    )
    logits = out.logits[:, -T - 1 : -1, :]
    if temperature != 1.0:
        logits = logits / temperature
    return gather_logprobs(logits, input_ids[:, -T:])


def masked_sum(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (values * mask).sum(-1)
