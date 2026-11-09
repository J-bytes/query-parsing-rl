"""Verifiable reward for query parsing: field F1 plus a format bonus.

Used in two places:
  * DPO: score k sampled completions per query, pick chosen / rejected.
  * GRPO: the scalar reward for each rollout.

No learned reward model. The reward compares the parse against the label, so it
is only as good as the labels; that is why the test set is hand-checked and why
the README reports an LLM-as-a-judge check that does not use the labels at all.

Reward definition (per completion):
    r = w_f1 * F1(pred, gold) + w_format * 1[strict JSON and schema-valid]
F1 is computed over (field, value) pairs. A wrong value counts as one false
positive and one false negative, the usual slot-filling convention. If both the
prediction and the label are empty, F1 = 1 (correctly predicting "nothing to
extract"). If nothing parses at all, F1 = 0 and the format term is 0.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from qprl.schema import parse_completion


@dataclass(frozen=True)
class PairCounts:
    tp: int
    fp: int
    fn: int


def count_pairs(pred: dict[str, Any], gold: dict[str, Any]) -> PairCounts:
    """TP/FP/FN over normalised (field, value) pairs."""
    tp = sum(1 for k, v in pred.items() if k in gold and gold[k] == v)
    return PairCounts(tp=tp, fp=len(pred) - tp, fn=len(gold) - tp)


def f1_from_counts(c: PairCounts) -> float:
    if c.tp + c.fp + c.fn == 0:
        return 1.0
    return 2 * c.tp / (2 * c.tp + c.fp + c.fn)


def example_f1(pred: dict[str, Any], gold: dict[str, Any]) -> float:
    return f1_from_counts(count_pairs(pred, gold))


@dataclass(frozen=True)
class RewardConfig:
    w_f1: float = 1.0
    w_format: float = 0.2


@dataclass(frozen=True)
class RewardBreakdown:
    total: float
    f1: float
    format_ok: bool
    lenient_json: bool


def score_completion(
    completion: str, gold: dict[str, Any], cfg: RewardConfig | None = None
) -> RewardBreakdown:
    """Reward for one completion against a normalised gold parse."""
    cfg = cfg or RewardConfig()
    res = parse_completion(completion)
    f1 = example_f1(res.parse, gold) if res.lenient_json else 0.0
    total = cfg.w_f1 * f1 + cfg.w_format * float(res.format_ok)
    return RewardBreakdown(
        total=total, f1=f1, format_ok=res.format_ok, lenient_json=res.lenient_json
    )
