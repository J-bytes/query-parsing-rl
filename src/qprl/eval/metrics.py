"""Corpus-level metrics for query parsing. One harness for every stage.

Reported numbers:
  * micro precision / recall / F1 over (field, value) pairs, pooled over the set
  * per-field precision / recall / F1 and support (number of gold occurrences)
  * strict valid-JSON rate: completion is exactly one JSON object
  * lenient valid-JSON rate: a JSON object can be recovered from the text
  * schema-valid rate: strict JSON with only known keys and correct types
  * exact match: normalised prediction equals normalised gold (empty == empty counts)
  * mean per-example F1 (the quantity the RL reward optimises, minus the format term)

Field-level metrics use the lenient parse, so a model is not punished twice for
a wrapper error; the wrapper error shows up in the strict JSON rate instead.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

from qprl.reward import PairCounts, count_pairs, example_f1
from qprl.schema import FIELDS, parse_completion


@dataclass
class FieldScore:
    precision: float
    recall: float
    f1: float
    support: int


@dataclass
class EvalReport:
    n: int
    micro_precision: float
    micro_recall: float
    micro_f1: float
    mean_example_f1: float
    exact_match: float
    strict_json_rate: float
    lenient_json_rate: float
    schema_valid_rate: float
    per_field: dict[str, FieldScore] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def evaluate(completions: Sequence[str], golds: Sequence[dict[str, Any]]) -> EvalReport:
    """Score raw completions against normalised gold parses."""
    if len(completions) != len(golds):
        raise ValueError(f"{len(completions)} completions vs {len(golds)} golds")
    n = len(completions)
    tp = fp = fn = 0
    per = {k: [0, 0, 0] for k in FIELDS}  # tp, fp, fn
    strict = lenient = schema = exact = 0
    f1_sum = 0.0
    for text, gold in zip(completions, golds, strict=True):
        res = parse_completion(text)
        pred = res.parse
        strict += res.strict_json
        lenient += res.lenient_json
        schema += res.format_ok
        exact += pred == gold
        f1_sum += example_f1(pred, gold)
        c: PairCounts = count_pairs(pred, gold)
        tp, fp, fn = tp + c.tp, fp + c.fp, fn + c.fn
        for k in FIELDS:
            in_p, in_g = k in pred, k in gold
            if in_p and in_g and pred[k] == gold[k]:
                per[k][0] += 1
            else:
                per[k][1] += in_p
                per[k][2] += in_g
    p, r, f = _prf(tp, fp, fn)
    per_field = {}
    for k, (ftp, ffp, ffn) in per.items():
        fp_, fr_, ff_ = _prf(ftp, ffp, ffn)
        per_field[k] = FieldScore(precision=fp_, recall=fr_, f1=ff_, support=ftp + ffn)
    denom = max(n, 1)
    return EvalReport(
        n=n,
        micro_precision=p,
        micro_recall=r,
        micro_f1=f,
        mean_example_f1=f1_sum / denom,
        exact_match=exact / denom,
        strict_json_rate=strict / denom,
        lenient_json_rate=lenient / denom,
        schema_valid_rate=schema / denom,
        per_field=per_field,
    )
