"""GRPO from scratch: group-relative advantages and the clipped surrogate loss.

Group Relative Policy Optimization (Shao et al., 2024, DeepSeekMath) is PPO
without a value network. For each prompt q we sample a *group* of G completions
o_1..o_G from the current policy, score each with a reward r_i, and use the
group itself as the baseline:

    A_i = (r_i - mean(r_1..r_G)) / (std(r_1..r_G) + eps)

Every token of completion i gets the same advantage A_i. The policy is then
updated with PPO's clipped importance-ratio objective, per token t:

    ratio_{i,t} = pi_theta(o_{i,t} | q, o_{i,<t}) / pi_old(o_{i,t} | q, o_{i,<t})
    L_{i,t}     = -min(ratio * A_i, clip(ratio, 1 - eps_low, 1 + eps_high) * A_i)
                  + beta * KL_{i,t}(pi_theta || pi_ref)

with the k3 KL estimator (Schulman, "Approximating KL divergence"), which is
unbiased and always >= 0:

    KL_{i,t} = exp(ref - logp) - (ref - logp) - 1

pi_old is the policy that generated the samples. With one optimisation pass
per batch (num_iterations = 1) pi_old == pi_theta numerically, the ratio is 1
and the clip never fires; only the gradient of the ratio matters, and
d ratio / d theta = d log pi_theta / d theta, so the loss reduces to REINFORCE
with a group baseline. Clipping only matters when we reuse samples for
num_iterations > 1 updates.

How per-token losses become one scalar is the main design choice, and it is
where GRPO variants differ:
    "grpo"    mean over tokens of each sequence, then mean over sequences
              (original paper). Short sequences get larger per-token weight,
              so the model is pushed to make wrong answers *longer*.
    "bnpo"    sum over all tokens / number of tokens in this micro-batch.
    "dapo"    sum over all tokens / number of tokens in the whole generation
              batch (DAPO, Yu et al. 2025). No length bias, and the scale does
              not depend on how the batch is split into micro-batches.
    "dr_grpo" sum over all tokens / (num sequences * max_completion_length)
              (Dr. GRPO, Liu et al. 2025): a constant normaliser, no length bias.
Dr. GRPO also drops the std division in the advantage (scale="none"), because
dividing by a small group std up-weights prompts that are almost solved or
almost hopeless.

Shapes used below: N = number of completions in the batch (= prompts * G),
T = completion length (right-padded), mask is 1 on real completion tokens.

This file has no model code on purpose: every function is a pure tensor
function, unit-tested on hand-built tensors and cross-checked against TRL's
GRPOTrainer._compute_loss in `grpo/trl_check.py` and `tests/test_trl_crosscheck.py`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

AdvantageScale = Literal["group", "batch", "none"]
LossAggregation = Literal["grpo", "bnpo", "dapo", "dr_grpo"]

# Same epsilon as TRL, so advantages match exactly in the cross-check.
ADV_EPS = 1e-4


def _unbiased_std(x: torch.Tensor, dim: int | None = None) -> torch.Tensor:
    """Sample std with Bessel's correction. NaN for a single element."""
    if dim is None:
        return x.std(unbiased=True) if x.numel() > 1 else torch.tensor(float("nan"))
    return x.std(dim=dim, unbiased=True)


@dataclass
class Advantages:
    advantages: torch.Tensor  # (N,)
    group_mean: torch.Tensor  # (N,) mean reward of each row's group
    group_std: torch.Tensor  # (N,) std of each row's group
    zero_std_fraction: float  # share of groups where every completion got the same reward


def group_advantages(
    rewards: torch.Tensor, group_size: int, scale: AdvantageScale = "group"
) -> Advantages:
    """Group-relative advantages.

    rewards: (N,) with rows grouped by prompt: rows [k*G, (k+1)*G) share prompt k.

    scale="group": divide by each group's std (original GRPO).
    scale="batch": divide by the std of the whole batch (more stable when many
                   groups have tiny std; recommended by "Tricks or Traps", 2025).
    scale="none":  no division (Dr. GRPO).

    A group whose G rewards are all equal has zero advantage for every member,
    so it contributes no policy gradient (only the KL term). The fraction of such
    groups is the single most useful GRPO health metric: if it approaches 1, the
    task is too easy or too hard for the current policy and learning stalls.
    """
    if rewards.dim() != 1 or rewards.numel() % group_size != 0:
        raise ValueError(f"rewards must be (N,) with N divisible by G={group_size}")
    grouped = rewards.view(-1, group_size)  # (num_prompts, G)
    mean = grouped.mean(dim=1).repeat_interleave(group_size)
    std_g = (
        _unbiased_std(grouped, dim=1)
        if group_size > 1
        else torch.full((grouped.size(0),), float("nan"), dtype=rewards.dtype)
    )
    std_rows = std_g.repeat_interleave(group_size)
    centred = rewards - mean
    if scale == "group":
        adv = centred / (std_rows + ADV_EPS)
    elif scale == "batch":
        adv = centred / (_unbiased_std(rewards) + ADV_EPS)
    elif scale == "none":
        adv = centred
    else:
        raise ValueError(f"unknown scale {scale!r}")
    adv = torch.nan_to_num(adv, nan=0.0)
    zero_std = torch.isclose(std_g, torch.zeros_like(std_g)).float().mean().item()
    return Advantages(
        advantages=adv, group_mean=mean, group_std=std_rows, zero_std_fraction=zero_std
    )


def k3_kl(logp: torch.Tensor, ref_logp: torch.Tensor) -> torch.Tensor:
    """Per-token k3 estimator of KL(pi_theta || pi_ref). Elementwise, >= 0.

    Samples come from pi_theta (well, pi_old ~= pi_theta), and with
    x = ref_logp - logp the estimator is e^x - x - 1, which is >= 0 because
    e^x >= 1 + x, and its expectation under pi_theta is the true KL.
    """
    x = ref_logp - logp
    return torch.exp(x) - x - 1.0


@dataclass
class GRPOLossOutput:
    loss: torch.Tensor  # scalar to backpropagate
    policy_loss: torch.Tensor  # scalar, detached: surrogate term only
    mean_kl: torch.Tensor  # scalar, detached, token-mean over mask
    clip_low_fraction: torch.Tensor  # tokens where A<0 and ratio < 1-eps_low
    clip_high_fraction: torch.Tensor  # tokens where A>0 and ratio > 1+eps_high
    mean_ratio: torch.Tensor


def aggregate(
    per_token_loss: torch.Tensor,
    mask: torch.Tensor,
    mode: LossAggregation,
    max_completion_length: int | None = None,
    num_items_in_batch: float | None = None,
) -> torch.Tensor:
    """Reduce (N, T) per-token losses to a scalar. See module docstring."""
    m = mask.to(per_token_loss.dtype)
    if mode == "grpo":
        return ((per_token_loss * m).sum(-1) / m.sum(-1).clamp(min=1.0)).mean()
    if mode == "bnpo":
        return (per_token_loss * m).sum() / m.sum().clamp(min=1.0)
    if mode == "dapo":
        denom = num_items_in_batch if num_items_in_batch is not None else m.sum().item()
        return (per_token_loss * m).sum() / max(float(denom), 1.0)
    if mode == "dr_grpo":
        if max_completion_length is None:
            raise ValueError("dr_grpo needs max_completion_length")
        return (per_token_loss * m).sum() / (per_token_loss.size(0) * max_completion_length)
    raise ValueError(f"unknown aggregation {mode!r}")


def grpo_loss(
    logp: torch.Tensor,
    old_logp: torch.Tensor | None,
    ref_logp: torch.Tensor | None,
    advantages: torch.Tensor,
    mask: torch.Tensor,
    epsilon_low: float = 0.2,
    epsilon_high: float = 0.2,
    beta: float = 0.0,
    aggregation: LossAggregation = "dapo",
    max_completion_length: int | None = None,
    num_items_in_batch: float | None = None,
    kl_ratio_correction: bool = False,
) -> GRPOLossOutput:
    """Clipped GRPO surrogate plus optional k3 KL penalty.

    logp:       (N, T) log pi_theta of the sampled tokens, WITH grad.
    old_logp:   (N, T) log pi_old (no grad). None means "same as logp" (the
                num_iterations=1 case): we use logp.detach(), so the ratio is
                exactly 1 but its gradient is grad log pi_theta.
    ref_logp:   (N, T) log pi_ref (no grad). Required if beta > 0.
    advantages: (N,) one scalar per completion, broadcast over its tokens.
    mask:       (N, T) 1 on real completion tokens.
    kl_ratio_correction: multiply the per-token KL by the importance ratio
                pi_theta / pi_old. The k3 estimate is unbiased only for samples
                drawn from pi_theta; ours come from pi_old, and the ratio corrects
                for that once num_iterations > 1. Off in the original paper, on by
                default in TRL >= 1.x (`use_bias_correction_kl`). With
                num_iterations = 1 the ratio is exactly 1 and this changes nothing
                in the value, only adds the ratio's gradient to the KL term.
    """
    if old_logp is None:
        old_logp = logp.detach()
    adv = advantages.unsqueeze(1)  # (N, 1) broadcast over tokens
    ratio = torch.exp(logp - old_logp)
    clipped = torch.clamp(ratio, 1.0 - epsilon_low, 1.0 + epsilon_high)
    # Pessimistic bound: take the smaller objective, i.e. the larger loss.
    per_token = -torch.min(ratio * adv, clipped * adv)
    policy_only = per_token
    m = mask.to(per_token.dtype)
    if beta != 0.0:
        if ref_logp is None:
            raise ValueError("beta > 0 needs ref_logp")
        kl = k3_kl(logp, ref_logp)
        kl_term = kl * ratio if kl_ratio_correction else kl
        per_token = per_token + beta * kl_term
        mean_kl = ((kl * m).sum() / m.sum().clamp(min=1.0)).detach()
    else:
        mean_kl = torch.zeros((), device=logp.device)
    loss = aggregate(per_token, mask, aggregation, max_completion_length, num_items_in_batch)
    policy_loss = aggregate(
        policy_only, mask, aggregation, max_completion_length, num_items_in_batch
    ).detach()
    with torch.no_grad():
        n_tok = m.sum().clamp(min=1.0)
        low = ((ratio < 1.0 - epsilon_low) & (adv < 0)).to(m.dtype)
        high = ((ratio > 1.0 + epsilon_high) & (adv > 0)).to(m.dtype)
        clip_low = (low * m).sum() / n_tok
        clip_high = (high * m).sum() / n_tok
        mean_ratio = (ratio * m).sum() / n_tok
    return GRPOLossOutput(
        loss=loss,
        policy_loss=policy_loss,
        mean_kl=mean_kl,
        clip_low_fraction=clip_low,
        clip_high_fraction=clip_high,
        mean_ratio=mean_ratio,
    )
