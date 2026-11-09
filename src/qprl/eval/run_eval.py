"""Evaluate one model on one split with the shared harness.

    make eval MODEL=runs/sft/best NAME=sft
    uv run --extra gpu python -m qprl.eval.run_eval --model runs/sft/best --name sft

Writes:
  results/<name>/<split>_metrics.json      all metrics
  results/<name>/<split>_predictions.jsonl one row per query: completion, parse, gold, F1

Greedy decoding, so results are deterministic for a given checkpoint. On the
test split only rows marked `reviewed` (and not `excluded`) are scored if any
row is reviewed; otherwise all rows are scored and a warning is printed. The
metrics file records which.

`--rules` evaluates the rule labeller instead of a model (baseline row).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from qprl.data.rules import RuleLabeller
from qprl.generation import generate_in_batches
from qprl.reward import example_f1
from qprl.schema import normalize_parse, parse_completion, to_target_json
from qprl.tasks import make_task
from qprl.utils import load_model_and_tokenizer, pick_device, read_jsonl, write_jsonl


def select_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], str]:
    if any("reviewed" in r for r in rows):
        reviewed = [r for r in rows if r.get("reviewed") and not r.get("excluded")]
        if reviewed:
            return reviewed, f"reviewed ({len(reviewed)}/{len(rows)})"
        print("WARNING: no test rows reviewed yet; scoring unreviewed LLM labels")
        return rows, "unreviewed"
    return rows, "all"


def run(
    model_path: str | None,
    data: Path,
    out_dir: Path,
    task_name: str,
    batch_size: int,
    max_new_tokens: int,
    dtype: str,
    limit: int | None,
    rules_brands: Path | None = None,
) -> dict[str, Any]:
    task = make_task(task_name)
    rows, subset = select_rows(read_jsonl(data))
    if limit:
        rows = rows[:limit]
    if model_path is None:  # rule baseline
        rl = RuleLabeller.from_file(rules_brands)
        texts = [to_target_json(rl.label(r["query"])) for r in rows]
    else:
        model, tok = load_model_and_tokenizer(model_path, dtype)
        model.to(pick_device())
        prompts = [task.prompt(tok, r) for r in rows]
        texts = generate_in_batches(model, tok, prompts, batch_size, max_new_tokens)
    metrics = task.evaluate(texts, rows)
    metrics["subset"] = subset
    metrics["model"] = model_path or "rules"
    out_dir.mkdir(parents=True, exist_ok=True)
    split = data.stem
    (out_dir / f"{split}_metrics.json").write_text(json.dumps(metrics, indent=2))
    preds = []
    for r, t in zip(rows, texts, strict=True):
        rec: dict[str, Any] = {"id": r["id"], "completion": t}
        if task_name == "query_parsing":
            p = parse_completion(t)
            gold = normalize_parse(r["label"])
            rec |= {
                "query": r["query"], "gold": gold, "pred": p.parse,
                "strict_json": p.strict_json, "f1": example_f1(p.parse, gold),
            }  # fmt: skip
        else:
            rec |= {"question": r["question"], "answer": r["answer"]}
        preds.append(rec)
    write_jsonl(out_dir / f"{split}_predictions.jsonl", preds)
    headline = {k: v for k, v in metrics.items() if isinstance(v, float)}
    print(json.dumps(headline, indent=2))
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--model", help="HF checkpoint dir or hub id")
    ap.add_argument("--rules", action="store_true", help="evaluate the rule baseline")
    ap.add_argument("--name", required=True, help="results/<name>/")
    ap.add_argument("--data", type=Path, default=Path("data/processed/test.jsonl"))
    ap.add_argument("--task", default="query_parsing")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--max-new-tokens", type=int, default=96)
    ap.add_argument("--dtype", default="bf16")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--brands", type=Path, default=Path("data/interim/brands.txt"))
    ap.add_argument("--results-dir", type=Path, default=Path("results"))
    args = ap.parse_args()
    if not args.rules and not args.model:
        raise SystemExit("pass --model or --rules")
    run(
        None if args.rules else args.model,
        args.data,
        args.results_dir / args.name,
        args.task,
        args.batch_size,
        args.max_new_tokens,
        args.dtype,
        args.limit,
        args.brands,
    )


if __name__ == "__main__":
    main()
