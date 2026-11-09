from __future__ import annotations

import math

import pytest
import torch
from hypothesis import given
from hypothesis import strategies as st

from qprl.dpo.loss import dpo_loss


def _t(*x: float) -> torch.Tensor:
    return torch.tensor(x, dtype=torch.float64)


def test_policy_equals_reference_gives_log2() -> None:
    lp = _t(-5.0, -12.0, -3.0)
    out = dpo_loss(lp, lp - 1, lp, lp - 1, beta=0.1)
    assert out.loss.item() == pytest.approx(math.log(2))
    assert out.reward_accuracy.item() == 0.0  # ties are not "correct"
    assert out.reward_margin.item() == pytest.approx(0.0)


def test_known_value() -> None:
    # chosen log-ratio +2, rejected log-ratio -1, beta 0.5 -> z = 1.5
    out = dpo_loss(_t(-8.0), _t(-11.0), _t(-10.0), _t(-10.0), beta=0.5)
    assert out.loss.item() == pytest.approx(-math.log(1 / (1 + math.exp(-1.5))))
    assert out.chosen_rewards.item() == pytest.approx(1.0)
    assert out.rejected_rewards.item() == pytest.approx(-0.5)
    assert out.reward_accuracy.item() == 1.0


def test_label_smoothing_value() -> None:
    z = 0.1 * (2.0 - (-1.0))
    out = dpo_loss(_t(-8.0), _t(-11.0), _t(-10.0), _t(-10.0), beta=0.1, label_smoothing=0.2)
    expected = -0.8 * math.log(1 / (1 + math.exp(-z))) - 0.2 * math.log(1 / (1 + math.exp(z)))
    assert out.loss.item() == pytest.approx(expected)


def test_label_smoothing_bounds() -> None:
    with pytest.raises(ValueError, match="label_smoothing"):
        dpo_loss(_t(0.0), _t(0.0), _t(0.0), _t(0.0), beta=0.1, label_smoothing=0.5)


def test_gradient_direction() -> None:
    """Gradient descent raises chosen log-prob and lowers rejected log-prob."""
    pc = _t(-5.0).requires_grad_()
    pr = _t(-5.0).requires_grad_()
    dpo_loss(pc, pr, _t(-5.0), _t(-5.0), beta=0.1).loss.backward()
    assert pc.grad is not None and pr.grad is not None
    assert pc.grad.item() < 0 < pr.grad.item()
    assert pc.grad.item() == pytest.approx(-pr.grad.item())


def test_reference_has_no_grad_path() -> None:
    rc = _t(-5.0).requires_grad_()
    out = dpo_loss(_t(-4.0), _t(-6.0), rc.detach(), _t(-5.0), beta=0.1)
    assert out.loss.grad_fn is None or rc.grad is None


@given(
    st.floats(-50, 0), st.floats(-50, 0), st.floats(-50, 0), st.floats(-50, 0), st.floats(0.01, 2)
)
def test_loss_positive_and_shift_invariant(
    a: float, b: float, c: float, d: float, beta: float
) -> None:
    out = dpo_loss(_t(a), _t(b), _t(c), _t(d), beta)
    assert out.loss.item() > 0
    # Adding the same constant to a sequence's policy and reference log-prob cancels.
    shifted = dpo_loss(_t(a + 3), _t(b - 2), _t(c + 3), _t(d - 2), beta)
    assert shifted.loss.item() == pytest.approx(out.loss.item(), rel=1e-9, abs=1e-12)


def test_toy_dpo_learns_preference() -> None:
    """Categorical policy over 4 'completions'; DPO on pair (0 > 3) moves mass to 0."""
    torch.manual_seed(0)
    logits = torch.zeros(4, requires_grad=True)
    ref = torch.log_softmax(torch.zeros(4), -1).detach()
    opt = torch.optim.SGD([logits], lr=1.0)
    for _ in range(50):
        lp = torch.log_softmax(logits, -1)
        loss = dpo_loss(lp[:1], lp[3:], ref[:1], ref[3:], beta=0.5).loss
        opt.zero_grad()
        loss.backward()
        opt.step()
    p = torch.softmax(logits, -1)
    assert p[0] > 0.25 > p[3]
