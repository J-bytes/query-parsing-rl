from __future__ import annotations

import math

import pytest
import torch
from hypothesis import given
from hypothesis import strategies as st

from qprl.grpo.algorithm import ADV_EPS, aggregate, group_advantages, grpo_loss, k3_kl


def test_group_advantages_hand_computed() -> None:
    r = torch.tensor([1.0, 0.0, 0.0, 1.0, 0.5, 0.5, 0.5, 0.5])
    a = group_advantages(r, group_size=4, scale="group")
    std = torch.tensor([1.0, 0.0, 0.0, 1.0]).std(unbiased=True).item()
    expected0 = 0.5 / (std + ADV_EPS)
    assert a.advantages[:4].tolist() == pytest.approx(
        [expected0, -expected0, -expected0, expected0]
    )
    assert a.advantages[4:].tolist() == [0.0] * 4  # all-equal group: no signal
    assert a.zero_std_fraction == 0.5
    assert a.group_mean.tolist() == [0.5] * 8


def test_scale_none_and_batch() -> None:
    r = torch.tensor([2.0, 0.0, 1.0, 1.0])
    assert group_advantages(r, 2, "none").advantages.tolist() == [1.0, -1.0, 0.0, 0.0]
    b = group_advantages(r, 2, "batch").advantages
    assert b.tolist() == pytest.approx(
        (torch.tensor([1.0, -1.0, 0, 0]) / (r.std() + ADV_EPS)).tolist()
    )


def test_bad_shapes() -> None:
    with pytest.raises(ValueError, match="divisible"):
        group_advantages(torch.zeros(5), 2)
    with pytest.raises(ValueError, match="unknown scale"):
        group_advantages(torch.zeros(4), 2, "nope")  # type: ignore[arg-type]


def test_group_size_one_gives_zero_advantage() -> None:
    a = group_advantages(torch.tensor([1.0, 2.0]), 1)
    assert a.advantages.tolist() == [0.0, 0.0]


@given(st.lists(st.floats(-5, 5), min_size=8, max_size=8))
def test_advantages_zero_mean_per_group(vals: list[float]) -> None:
    a = group_advantages(torch.tensor(vals, dtype=torch.float64), 4, "group").advantages
    for g in range(2):
        assert a[g * 4 : (g + 1) * 4].sum().item() == pytest.approx(0.0, abs=1e-6)


@given(st.floats(-20, 0), st.floats(-20, 0))
def test_k3_nonnegative_and_zero_at_equality(lp: float, ref: float) -> None:
    v = k3_kl(torch.tensor(lp, dtype=torch.float64), torch.tensor(ref, dtype=torch.float64)).item()
    assert v >= -1e-12
    assert k3_kl(torch.tensor(lp), torch.tensor(lp)).item() == 0.0


def test_k3_is_unbiased_kl_estimate() -> None:
    """E_{x~p}[k3] equals KL(p||q) for categorical p, q."""
    p = torch.tensor([0.5, 0.3, 0.2], dtype=torch.float64)
    q = torch.tensor([0.2, 0.3, 0.5], dtype=torch.float64)
    est = (p * k3_kl(p.log(), q.log())).sum().item()
    exact = (p * (p / q).log()).sum().item()
    assert est == pytest.approx(exact)


def test_on_policy_loss_value_and_gradient() -> None:
    """old_logp=None -> ratio 1, loss = -mean token advantage, grad = -A * dlogp."""
    logp = torch.tensor([[-1.0, -2.0], [-0.5, -0.5]], requires_grad=True)
    adv = torch.tensor([1.0, -2.0])
    mask = torch.tensor([[1, 1], [1, 0]])
    out = grpo_loss(logp, None, None, adv, mask, aggregation="bnpo")
    assert out.loss.item() == pytest.approx(-(1 + 1 - 2) / 3)
    out.loss.backward()
    assert logp.grad is not None
    assert logp.grad.flatten().tolist() == pytest.approx([-1 / 3, -1 / 3, 2 / 3, 0.0])
    assert out.mean_ratio.item() == pytest.approx(1.0)
    assert out.clip_low_fraction.item() == 0.0 == out.clip_high_fraction.item()


def test_clipping_stops_gradient_beyond_trust_region() -> None:
    # ratio = e^0.5 ~ 1.65 > 1.2 with positive advantage: clipped, zero grad.
    logp = torch.tensor([[0.0]], requires_grad=True)
    old = torch.tensor([[-0.5]])
    out = grpo_loss(logp, old, None, torch.tensor([1.0]), torch.ones(1, 1), aggregation="bnpo")
    assert out.loss.item() == pytest.approx(-1.2)
    out.loss.backward()
    assert logp.grad is not None and logp.grad.item() == 0.0
    assert out.clip_high_fraction.item() == 1.0


def test_clipping_does_not_apply_when_it_would_help() -> None:
    # ratio 1.65 with NEGATIVE advantage: min picks unclipped -> gradient flows.
    logp = torch.tensor([[0.0]], requires_grad=True)
    out = grpo_loss(logp, torch.tensor([[-0.5]]), None, torch.tensor([-1.0]), torch.ones(1, 1),
                    aggregation="bnpo")  # fmt: skip
    assert out.loss.item() == pytest.approx(math.exp(0.5))
    out.loss.backward()
    assert logp.grad is not None and logp.grad.item() == pytest.approx(math.exp(0.5))


def test_asymmetric_clip() -> None:
    logp = torch.tensor([[0.0]])
    old = torch.tensor([[-0.22]])  # ratio ~ 1.246: above 1.2, below 1.28
    lo = grpo_loss(
        logp, old, None, torch.tensor([1.0]), torch.ones(1, 1), 0.2, 0.2, aggregation="bnpo"
    )
    hi = grpo_loss(
        logp, old, None, torch.tensor([1.0]), torch.ones(1, 1), 0.2, 0.28, aggregation="bnpo"
    )
    assert lo.loss.item() == pytest.approx(-1.2)
    assert hi.loss.item() == pytest.approx(-math.exp(0.22))


def test_kl_term_and_ratio_correction() -> None:
    logp = torch.tensor([[-1.0]])
    ref = torch.tensor([[-1.5]])
    old = torch.tensor([[-1.2]])
    kl = math.exp(-0.5) + 0.5 - 1
    plain = grpo_loss(
        logp, old, ref, torch.tensor([0.0]), torch.ones(1, 1), beta=0.1, aggregation="bnpo"
    )
    assert plain.loss.item() == pytest.approx(0.1 * kl)
    assert plain.mean_kl.item() == pytest.approx(kl)
    corr = grpo_loss(logp, old, ref, torch.tensor([0.0]), torch.ones(1, 1), beta=0.1,
                     aggregation="bnpo", kl_ratio_correction=True)  # fmt: skip
    assert corr.loss.item() == pytest.approx(0.1 * kl * math.exp(0.2))
    with pytest.raises(ValueError, match="ref_logp"):
        grpo_loss(logp, None, None, torch.tensor([0.0]), torch.ones(1, 1), beta=0.1)


def test_aggregations() -> None:
    per_tok = torch.tensor([[1.0, 1.0, 1.0, 1.0], [4.0, 0.0, 0.0, 0.0]])
    mask = torch.tensor([[1, 1, 1, 1], [1, 0, 0, 0]])
    assert aggregate(per_tok, mask, "grpo").item() == pytest.approx((1.0 + 4.0) / 2)
    assert aggregate(per_tok, mask, "bnpo").item() == pytest.approx(8.0 / 5)
    assert aggregate(per_tok, mask, "dapo").item() == pytest.approx(8.0 / 5)
    assert aggregate(per_tok, mask, "dapo", num_items_in_batch=10).item() == pytest.approx(0.8)
    assert aggregate(per_tok, mask, "dr_grpo", max_completion_length=8).item() == pytest.approx(
        8 / 16
    )
    with pytest.raises(ValueError, match="max_completion_length"):
        aggregate(per_tok, mask, "dr_grpo")
    with pytest.raises(ValueError, match="unknown aggregation"):
        aggregate(per_tok, mask, "x")  # type: ignore[arg-type]


def test_grpo_length_bias_vs_dapo() -> None:
    """Same advantage per token: 'grpo' weights a short sequence's tokens more."""
    logp = torch.zeros(2, 4, requires_grad=True)
    mask = torch.tensor([[1, 1, 1, 1], [1, 0, 0, 0]])
    grpo_loss(logp, None, None, torch.tensor([1.0, 1.0]), mask, aggregation="grpo").loss.backward()
    assert logp.grad is not None
    assert abs(logp.grad[1, 0].item()) == pytest.approx(4 * abs(logp.grad[0, 0].item()))
    logp2 = torch.zeros(2, 4, requires_grad=True)
    grpo_loss(logp2, None, None, torch.tensor([1.0, 1.0]), mask, aggregation="dapo").loss.backward()
    assert logp2.grad is not None
    assert logp2.grad[1, 0].item() == pytest.approx(logp2.grad[0, 0].item())


def test_toy_bandit_learns() -> None:
    """End-to-end GRPO on a 5-armed bandit: the best arm's probability rises."""
    torch.manual_seed(0)
    arm_reward = torch.tensor([0.0, 0.2, 1.0, 0.1, 0.0])
    logits = torch.zeros(5, requires_grad=True)
    ref_logits = torch.zeros(5)
    opt = torch.optim.Adam([logits], lr=0.1)
    G, P = 8, 4
    for _ in range(60):
        with torch.no_grad():
            arms = torch.multinomial(torch.softmax(logits, -1), P * G, replacement=True)
        rewards = arm_reward[arms] + 0.05 * torch.randn(P * G)
        adv = group_advantages(rewards, G).advantages
        old = torch.log_softmax(logits.detach(), -1)[arms].unsqueeze(1)
        for _ in range(2):  # num_iterations = 2: clipping active on 2nd pass
            lp = torch.log_softmax(logits, -1)[arms].unsqueeze(1)
            ref = torch.log_softmax(ref_logits, -1)[arms].unsqueeze(1)
            out = grpo_loss(lp, old, ref, adv, torch.ones_like(lp), beta=0.01)
            opt.zero_grad()
            out.loss.backward()
            opt.step()
    assert torch.softmax(logits, -1).argmax().item() == 2
    assert torch.softmax(logits, -1)[2].item() > 0.6


# --- tests added after mutation testing (mutmut) exposed untested paths ---


def test_lower_clip_with_negative_advantage() -> None:
    # ratio = e^-0.5 ~ 0.61 < 0.8 and A = -2: clipped at 0.8, gradient blocked.
    logp = torch.tensor([[-0.5]], requires_grad=True)
    out = grpo_loss(logp, torch.tensor([[0.0]]), None, torch.tensor([-2.0]), torch.ones(1, 1),
                    epsilon_low=0.2, epsilon_high=0.28, aggregation="bnpo")  # fmt: skip
    assert out.loss.item() == pytest.approx(0.8 * 2.0)
    assert out.clip_low_fraction.item() == 1.0 and out.clip_high_fraction.item() == 0.0
    out.loss.backward()
    assert logp.grad is not None and logp.grad.item() == 0.0


def test_unclipped_value_uses_product_with_advantage() -> None:
    logp = torch.tensor([[0.0]])
    old = torch.tensor([[-0.1]])  # ratio ~ 1.105, inside the trust region
    out = grpo_loss(logp, old, None, torch.tensor([3.0]), torch.ones(1, 1), aggregation="bnpo")
    assert out.loss.item() == pytest.approx(-3.0 * math.exp(0.1))
    assert out.policy_loss.item() == pytest.approx(out.loss.item())
    assert out.mean_ratio.item() == pytest.approx(math.exp(0.1))


def test_clip_fractions_and_kl_mean_over_tokens() -> None:
    logp = torch.zeros(2, 3)
    old = torch.tensor(
        [[-0.5, 0.0, 0.0], [0.5, 0.0, 0.0]]
    )  # row0 tok0 high ratio, row1 tok0 low ratio
    ref = torch.tensor([[-1.0, 0.0, -1.0], [0.0, 0.0, 0.0]])
    adv = torch.tensor([1.0, -1.0])
    mask = torch.tensor([[1, 1, 1], [1, 1, 0]])
    out = grpo_loss(logp, old, ref, adv, mask, beta=0.1, aggregation="bnpo")
    assert out.clip_high_fraction.item() == pytest.approx(1 / 5)
    assert out.clip_low_fraction.item() == pytest.approx(1 / 5)
    k = math.exp(-1) + 1 - 1  # k3 at ref - logp = -1
    assert out.mean_kl.item() == pytest.approx(2 * k / 5)
    # At exactly ratio == 1 + eps nothing counts as clipped.
    edge = grpo_loss(torch.zeros(1, 1), torch.full((1, 1), -math.log(1.2)), None,
                     torch.tensor([1.0]), torch.ones(1, 1), aggregation="bnpo")  # fmt: skip
    assert edge.clip_high_fraction.item() == 0.0


def test_zero_advantage_is_not_clip_counted() -> None:
    out = grpo_loss(torch.zeros(1, 1), torch.tensor([[-1.0]]), None, torch.tensor([0.0]),
                    torch.ones(1, 1), aggregation="bnpo")  # fmt: skip
    assert out.clip_high_fraction.item() == 0.0 and out.clip_low_fraction.item() == 0.0
    out2 = grpo_loss(torch.zeros(1, 1), torch.tensor([[1.0]]), None, torch.tensor([0.5]),
                     torch.ones(1, 1), aggregation="bnpo")  # fmt: skip
    assert out2.clip_low_fraction.item() == 0.0  # low ratio but positive advantage


def test_grpo_loss_passes_normalisers_through() -> None:
    logp = torch.zeros(1, 2)
    adv = torch.tensor([1.0])
    mask = torch.ones(1, 2)
    dr = grpo_loss(logp, None, None, adv, mask, aggregation="dr_grpo", max_completion_length=8)
    assert dr.loss.item() == pytest.approx(-2 / 8) and dr.policy_loss.item() == pytest.approx(
        -2 / 8
    )
    da = grpo_loss(logp, None, None, adv, mask, aggregation="dapo", num_items_in_batch=10)
    assert da.loss.item() == pytest.approx(-2 / 10) and da.policy_loss.item() == pytest.approx(-0.2)
    assert grpo_loss(logp, None, None, adv, mask).mean_kl.item() == 0.0


def test_dapo_small_denominator() -> None:
    per_tok = torch.tensor([[3.0]])
    assert aggregate(per_tok, torch.ones(1, 1), "dapo", num_items_in_batch=1).item() == 3.0


def test_group_std_reported() -> None:
    a = group_advantages(torch.tensor([0.0, 2.0, 1.0, 1.0]), 2)
    assert a.group_std.tolist() == pytest.approx([math.sqrt(2)] * 2 + [0.0] * 2)


def test_low_ratio_zero_advantage_not_clip_counted() -> None:
    out = grpo_loss(torch.zeros(1, 1), torch.tensor([[1.0]]), None, torch.tensor([0.0]),
                    torch.ones(1, 1), aggregation="bnpo")  # fmt: skip
    assert out.clip_low_fraction.item() == 0.0
