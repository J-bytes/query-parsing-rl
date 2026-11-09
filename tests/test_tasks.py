from __future__ import annotations

import pytest

from qprl.tasks import GSM8KTask, extract_gsm8k_answer, make_task
from qprl.training import warmup_cosine


def test_gsm8k_answer_extraction() -> None:
    assert extract_gsm8k_answer("so 3+4=7\n#### 7") == ("7", True)
    assert extract_gsm8k_answer("#### 1,234") == ("1234", True)
    assert extract_gsm8k_answer("the answer is 12.") == ("12", False)
    assert extract_gsm8k_answer("no numbers") == (None, False)


def test_gsm8k_reward() -> None:
    t = GSM8KTask(w_format=0.1)
    assert t.reward("#### 42", {"answer": "42"}) == {"reward": pytest.approx(1.1), "accuracy": 1.0}
    assert t.reward("42", {"answer": "42"})["reward"] == 0.0  # no marker: no credit
    assert t.reward("#### 41", {"answer": "42"})["reward"] == pytest.approx(0.1)
    ev = t.evaluate(["#### 42", "41"], [{"answer": "42"}, {"answer": "41"}])
    assert ev["accuracy"] == 1.0 and ev["format_rate"] == 0.5


def test_query_task_normalises_gold() -> None:
    t = make_task("query_parsing")
    row = {"query": "lamp under $50", "label": {"product_type": "Lamp", "price_max": 50}}
    assert t.target(row) == '{"product_type": "lamp", "price_max": 50}'
    assert t.reward(t.target(row), row)["reward"] == pytest.approx(1.2)
    with pytest.raises(ValueError, match="unknown task"):
        make_task("nope")


def test_warmup_cosine() -> None:
    assert warmup_cosine(0, 10, 100) == pytest.approx(0.1)
    assert warmup_cosine(9, 10, 100) == pytest.approx(1.0)
    assert warmup_cosine(100, 10, 100) == pytest.approx(0.1)
    assert warmup_cosine(55, 10, 100) == pytest.approx(0.55)
