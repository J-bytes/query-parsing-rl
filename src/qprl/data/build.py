"""Merge rule and LLM labels into final train / dev / test files.

Merge policy (each rule is a design decision worth defending):
  1. Start from the LLM label, normalised to the schema. Rows whose LLM output
     is not valid JSON are dropped and counted.
  2. Price: if the rules find a price with an explicit "$" or currency word, the
     rule value overrides the LLM. Regexes are more reliable than a 7B model on
     "under $35" vs "$35 gift card".
  3. Extractive guard: brand, color, material, size and compatible_with must
     appear verbatim (after normalisation) in the query. A value that does not
     is treated as a hallucination and dropped from the label. product_type must
     share at least one token with the query. This is what makes the strict
     metric (no stemming) fair: the label is always a substring of the input.
  4. Every decision is counted and written to data/processed/build_stats.json,
     so the README can report how much the guard removed.

Output rows: {"id", "query", "label": {...normalised parse...}, "label_source"}.
The test file additionally gets "reviewed": false until a human checks it with
`make review`.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from qprl.data.llm_label import cache_key
from qprl.data.rules import RuleLabeller, extract_price
from qprl.schema import STRING_FIELDS, normalize_parse, normalize_text, parse_completion
from qprl.utils import read_jsonl, write_jsonl

EXTRACTIVE_FIELDS = [f for f in STRING_FIELDS if f not in ("product_type", "gender", "age_group")]


def apply_guard(query: str, label: dict[str, Any], stats: Counter[str]) -> dict[str, Any]:
    q = normalize_text(query.replace("'", ""))
    q_tokens = set(q.split())
    out = dict(label)
    for f in EXTRACTIVE_FIELDS:
        if f in out and f" {out[f]} " not in f" {q} ":
            stats[f"guard_dropped_{f}"] += 1
            del out[f]
    if "product_type" in out and not (set(str(out["product_type"]).split()) & q_tokens):
        stats["guard_dropped_product_type"] += 1
        del out["product_type"]
    return out


def merge_one(query: str, llm_raw: str, stats: Counter[str]) -> dict[str, Any] | None:
    res = parse_completion(llm_raw)
    if not res.lenient_json:
        stats["llm_invalid_json"] += 1
        return None
    label = dict(res.parse)
    rule_price = normalize_parse(extract_price(query, require_currency=True))
    if rule_price:
        for k in ("price_min", "price_max"):
            if label.get(k) != rule_price.get(k):
                stats["rule_price_override"] += 1
                break
        label = {k: v for k, v in label.items() if k not in ("price_min", "price_max")}
        label.update(rule_price)
    label = apply_guard(query, label, stats)
    stats["kept"] += 1
    return label


def build(
    interim_dir: Path, cache_path: Path, out_dir: Path, model: str, brands_path: Path | None
) -> dict[str, Any]:
    cache = {r["key"]: r for r in read_jsonl(cache_path)} if cache_path.exists() else {}
    rules = RuleLabeller.from_file(brands_path)
    all_stats: dict[str, Any] = {}
    for split in ("sft", "dpo", "grpo", "dev", "test"):
        qpath = interim_dir / f"queries_{split}.jsonl"
        if not qpath.exists():
            continue
        stats: Counter[str] = Counter()
        rows: list[dict[str, Any]] = []
        for r in read_jsonl(qpath):
            rec = cache.get(cache_key(r["query"], model))
            if rec is None:
                stats["missing_llm_label"] += 1
                continue
            label = merge_one(r["query"], rec["raw"], stats)
            if label is None:
                continue
            row: dict[str, Any] = {
                "id": r["id"],
                "query": r["query"],
                "label": label,
                "label_source": f"llm:{model}+rules",
            }
            if split == "test":
                row["reviewed"] = False
                row["rule_label"] = normalize_parse(rules.label(r["query"]))
            rows.append(row)
        write_jsonl(out_dir / f"{split}.jsonl", rows)
        all_stats[split] = dict(stats)
        print(f"{split}: {len(rows)} rows  {dict(stats)}")
    (out_dir / "build_stats.json").write_text(json.dumps(all_stats, indent=2))
    return all_stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--interim-dir", type=Path, default=Path("data/interim"))
    ap.add_argument("--cache", type=Path, default=Path("data/llm_cache/labels.jsonl"))
    ap.add_argument("--out-dir", type=Path, default=Path("data/processed"))
    ap.add_argument("--model", required=True, help="labeller model name used in the cache")
    ap.add_argument("--brands", type=Path, default=Path("data/interim/brands.txt"))
    args = ap.parse_args()
    test_path = args.out_dir / "test.jsonl"
    if test_path.exists() and any(r.get("reviewed") for r in read_jsonl(test_path)):
        raise SystemExit(
            f"{test_path} has reviewed rows; refusing to overwrite hand-checked labels. "
            "Move it away first if you really want to rebuild."
        )
    build(args.interim_dir, args.cache, args.out_dir, args.model, args.brands)


if __name__ == "__main__":
    main()
