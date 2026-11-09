from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from qprl.reward import RewardConfig, count_pairs, example_f1, score_completion

GOLD = {"product_type": "case", "compatible_with": "iphone 12", "color": "red"}


def test_counts_wrong_value_is_fp_and_fn() -> None:
    c = count_pairs({"product_type": "case", "color": "blue"}, GOLD)
    assert (c.tp, c.fp, c.fn) == (1, 1, 2)


def test_f1_values() -> None:
    assert example_f1(GOLD, GOLD) == 1.0
    assert example_f1({}, {}) == 1.0
    assert example_f1({"brand": "x"}, {}) == 0.0
    assert example_f1({}, GOLD) == 0.0
    assert example_f1({"product_type": "case"}, GOLD) == pytest.approx(2 / 4)


def test_score_completion() -> None:
    r = score_completion(
        '{"product_type": "case", "compatible_with": "iphone 12", "color": "red"}', GOLD
    )
    assert r.f1 == 1.0 and r.format_ok and r.total == pytest.approx(1.2)
    fenced = score_completion('```{"product_type": "case"}```', GOLD)
    assert not fenced.format_ok and fenced.total == pytest.approx(0.5)
    junk = score_completion("hello", GOLD)
    assert junk.total == 0.0 and not junk.lenient_json


def test_reward_weights() -> None:
    cfg = RewardConfig(w_f1=2.0, w_format=0.0)
    assert score_completion('{"product_type": "case"}', GOLD, cfg).total == pytest.approx(1.0)


def test_empty_gold_empty_pred_gets_full_reward() -> None:
    assert score_completion("{}", {}).total == pytest.approx(1.2)


small = st.dictionaries(st.sampled_from(["brand", "color", "size", "material"]),
                        st.sampled_from(["a", "b", "c"]), max_size=4)  # fmt: skip


@given(small, small)
def test_f1_bounds_and_symmetry(a: dict[str, str], b: dict[str, str]) -> None:
    f = example_f1(a, b)
    assert 0.0 <= f <= 1.0
    assert f == pytest.approx(example_f1(b, a))
    assert (f == 1.0) == (a == b)


@given(small)
def test_adding_wrong_field_never_helps(gold: dict[str, str]) -> None:
    pred = dict(gold)
    before = example_f1(pred, gold)
    pred["product_type"] = "zzz"
    assert example_f1(pred, gold) <= before


def test_garbage_on_empty_gold_scores_zero() -> None:
    """Unparseable output must not get the 'correctly empty' F1 of 1."""
    r = score_completion("garbage", {})
    assert r.f1 == 0.0 and r.total == 0.0 and r.lenient_json is False
