"""Direct Preference Optimization loss (Rafailov et al., 2023).

DPO starts from the KL-regularised RL objective
    max_pi  E[r(x, y)] - beta * KL(pi || pi_ref)
whose optimum is pi*(y|x) ∝ pi_ref(y|x) exp(r(x, y) / beta). Inverting gives an
implicit reward r(x, y) = beta * log(pi(y|x) / pi_ref(y|x)) + const(x). Plugging
that into a Bradley-Terry preference model, the const(x) cancels between the
chosen and rejected completions, which leaves a plain classification loss:

    L = -log sigmoid( beta * [ (log pi(y_w|x) - log pi_ref(y_w|x))
                             - (log pi(y_l|x) - log pi_ref(y_l|x)) ] )

where log pi(y|x) is the *sum* of completion token log-probs. With
label_smoothing = eps we get conservative DPO (cDPO), which assumes each label
is flipped with probability eps:
    L = -(1 - eps) * log sigmoid(z) - eps * log sigmoid(-z)

At initialisation pi == pi_ref, so z = 0 and L = log 2 ≈ 0.693 for every pair.
That is the first sanity check in the TRL cross-check.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class DPOOutput:
    loss: torch.Tensor  # scalar, mean over pairs
    chosen_rewards: torch.Tensor  # (B,) beta * log-ratio of chosen, detached
    rejected_rewards: torch.Tensor  # (B,) detached
    reward_accuracy: torch.Tensor  # scalar: fraction with chosen reward > rejected reward
    reward_margin: torch.Tensor  # scalar: mean(chosen - rejected)


def dpo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    ref_chosen_logps: torch.Tensor,
    ref_rejected_logps: torch.Tensor,
    beta: float,
    label_smoothing: float = 0.0,
) -> DPOOutput:
    """All inputs are (B,) summed sequence log-probs. Reference values carry no grad."""
    if not 0.0 <= label_smoothing < 0.5:
        raise ValueError("label_smoothing must be in [0, 0.5)")
    chosen_logratios = policy_chosen_logps - ref_chosen_logps
    rejected_logratios = policy_rejected_logps - ref_rejected_logps
    z = beta * (chosen_logratios - rejected_logratios)
    losses = -(1.0 - label_smoothing) * F.logsigmoid(z) - label_smoothing * F.logsigmoid(-z)
    chosen_rewards = (beta * chosen_logratios).detach()
    rejected_rewards = (beta * rejected_logratios).detach()
    return DPOOutput(
        loss=losses.mean(),
        chosen_rewards=chosen_rewards,
        rejected_rewards=rejected_rewards,
        reward_accuracy=(chosen_rewards > rejected_rewards).float().mean(),
        reward_margin=(chosen_rewards - rejected_rewards).mean(),
    )
