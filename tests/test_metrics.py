from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from qprl.eval.metrics import evaluate
from qprl.schema import to_target_json

GOLDS: list[dict[str, object]] = [
    {"product_type": "case", "compatible_with": "iphone 12"},
    {"product_type": "lamp", "price_max": 50.0},
    {},
]


def test_perfect() -> None:
    rep = evaluate([to_target_json(g) for g in GOLDS], GOLDS)
    assert rep.micro_f1 == 1.0 and rep.exact_match == 1.0
    assert rep.strict_json_rate == 1.0 and rep.schema_valid_rate == 1.0
    assert rep.per_field["price_max"].support == 1
    assert rep.per_field["brand"].support == 0 and rep.per_field["brand"].f1 == 0.0


def test_mixed() -> None:
    preds = [
        '{"product_type": "case"}',  # 1 tp, 1 fn
        'Here: {"product_type": "lamp", "price_max": 40}',  # lenient; 1 tp, 1 fp, 1 fn
        "garbage",  # empty pred vs empty gold: exact match
    ]
    rep = evaluate(preds, GOLDS)
    # tp=2 fp=1 fn=2 -> P=2/3 R=1/2
    assert rep.micro_precision == pytest.approx(2 / 3)
    assert rep.micro_recall == pytest.approx(1 / 2)
    assert rep.micro_f1 == pytest.approx(2 * (2 / 3) * 0.5 / (2 / 3 + 0.5))
    assert rep.strict_json_rate == pytest.approx(1 / 3)
    assert rep.lenient_json_rate == pytest.approx(2 / 3)
    assert rep.exact_match == pytest.approx(1 / 3)
    pm = rep.per_field["price_max"]
    assert (pm.precision, pm.recall, pm.support) == (0.0, 0.0, 1)
    assert rep.per_field["product_type"].f1 == 1.0


def test_length_mismatch() -> None:
    with pytest.raises(ValueError, match="completions"):
        evaluate(["{}"], [])


@given(st.lists(st.sampled_from(GOLDS), min_size=1, max_size=10))
def test_identity_predictions_score_one(golds: list[dict[str, object]]) -> None:
    rep = evaluate([to_target_json(g) for g in golds], golds)
    assert rep.exact_match == 1.0 and rep.mean_example_f1 == 1.0
    if any(golds):
        assert rep.micro_f1 == 1.0


def test_field_counts_and_empty_predictions() -> None:
    golds: list[dict[str, object]] = [{"product_type": "a"}, {"product_type": "b"}]
    rep = evaluate(['{"product_type": "a"}', '{"product_type": "b"}'], golds)
    assert rep.per_field["product_type"].support == 2
    empty = evaluate(["{}", "{}"], golds)
    assert empty.micro_precision == 0.0 and empty.micro_recall == 0.0
    assert empty.per_field["product_type"].precision == 0.0
    # tp == fp: precision 1/2, not 0
    half = evaluate(['{"product_type": "a", "brand": "x"}'], [{"product_type": "a"}])
    assert half.micro_precision == pytest.approx(0.5)
    assert half.per_field["brand"].recall == 0.0


def test_per_field_false_positives_accumulate() -> None:
    golds: list[dict[str, object]] = [{}, {}, {"brand": "a"}]
    rep = evaluate(['{"brand": "x"}', '{"brand": "y"}', '{"brand": "a"}'], golds)
    assert rep.n == 3
    assert rep.per_field["brand"].precision == pytest.approx(1 / 3)
