"""Task definitions: how to render a prompt, score a completion, and evaluate.

The trainers only talk to a `Task`, which is what lets GRPO run on GSM8K as a
second benchmark with no trainer changes.

Example rows:
  query parsing: {"id", "query", "label": {...}}
  gsm8k:         {"id", "question", "answer": "<final number>"}
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from qprl.eval.metrics import evaluate
from qprl.prompts import render_prompt
from qprl.reward import RewardConfig, score_completion
from qprl.schema import normalize_parse, to_target_json


class Task(Protocol):
    name: str

    def prompt(self, tokenizer: Any, row: dict[str, Any]) -> str: ...

    def target(self, row: dict[str, Any]) -> str: ...

    def reward(self, completion: str, row: dict[str, Any]) -> dict[str, float]: ...

    def evaluate(
        self, completions: Sequence[str], rows: Sequence[dict[str, Any]]
    ) -> dict[str, Any]: ...


@dataclass
class QueryParsingTask:
    reward_cfg: RewardConfig
    name: str = "query_parsing"

    def prompt(self, tokenizer: Any, row: dict[str, Any]) -> str:
        return render_prompt(tokenizer, row["query"])

    def target(self, row: dict[str, Any]) -> str:
        return to_target_json(normalize_parse(row["label"]))

    def reward(self, completion: str, row: dict[str, Any]) -> dict[str, float]:
        r = score_completion(completion, normalize_parse(row["label"]), self.reward_cfg)
        return {"reward": r.total, "f1": r.f1, "format_ok": float(r.format_ok)}

    def evaluate(
        self, completions: Sequence[str], rows: Sequence[dict[str, Any]]
    ) -> dict[str, Any]:
        return evaluate(completions, [normalize_parse(r["label"]) for r in rows]).to_dict()


GSM8K_SYSTEM = (
    "Solve the math word problem. Reason step by step, then give the final answer "
    "as a number on the last line in the form '#### <number>'."
)
_GSM_ANSWER = re.compile(r"####\s*(-?[\d,]*\.?\d+)")
_LAST_NUMBER = re.compile(r"-?[\d,]*\.?\d+")


def extract_gsm8k_answer(text: str) -> tuple[str | None, bool]:
    """(answer, used_marker). Falls back to the last number in the text."""
    m = _GSM_ANSWER.search(text)
    if m:
        return m.group(1).replace(",", ""), True
    nums = _LAST_NUMBER.findall(text)
    return (nums[-1].replace(",", "") if nums else None), False


def _num_equal(a: str | None, b: str) -> bool:
    if a is None:
        return False
    try:
        return abs(float(a) - float(b)) < 1e-6
    except ValueError:
        return False


@dataclass
class GSM8KTask:
    w_format: float = 0.1
    name: str = "gsm8k"

    def prompt(self, tokenizer: Any, row: dict[str, Any]) -> str:
        msgs = [
            {"role": "system", "content": GSM8K_SYSTEM},
            {"role": "user", "content": row["question"]},
        ]
        text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        assert isinstance(text, str)
        return text

    def target(self, row: dict[str, Any]) -> str:
        return str(row.get("solution", f"#### {row['answer']}"))

    def reward(self, completion: str, row: dict[str, Any]) -> dict[str, float]:
        ans, marker = extract_gsm8k_answer(completion)
        correct = float(marker and _num_equal(ans, row["answer"]))
        return {"reward": correct + self.w_format * float(marker), "accuracy": correct}

    def evaluate(
        self, completions: Sequence[str], rows: Sequence[dict[str, Any]]
    ) -> dict[str, Any]:
        n = max(len(rows), 1)
        acc = sum(
            _num_equal(extract_gsm8k_answer(c)[0], r["answer"])
            for c, r in zip(completions, rows, strict=True)
        )
        fmt = sum(extract_gsm8k_answer(c)[1] for c in completions)
        return {"n": len(rows), "accuracy": acc / n, "format_rate": fmt / n}


def make_task(name: str, w_f1: float = 1.0, w_format: float = 0.2) -> Task:
    if name == "query_parsing":
        return QueryParsingTask(RewardConfig(w_f1=w_f1, w_format=w_format))
    if name == "gsm8k":
        return GSM8KTask(w_format=w_format)
    raise ValueError(f"unknown task {name!r}")
