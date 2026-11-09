"""Build the README results table and failure-case summary from results/.

    make results            # prints markdown, writes results/summary.md

Also writes a paired bootstrap 95% CI for the micro-F1 difference between
consecutive stages (base->SFT, SFT->DPO, SFT->GRPO), computed on per-query
TP/FP/FN, so a 0.5-point change on 300 queries is not over-claimed.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from qprl.reward import count_pairs
from qprl.schema import FIELDS
from qprl.utils import read_jsonl

ROWS = ["rules", "base", "sft", "dpo", "grpo", "sft_dpo_grpo"]
COLS = [
    ("micro_f1", "Field F1"),
    ("micro_precision", "P"),
    ("micro_recall", "R"),
    ("exact_match", "Exact match"),
    ("strict_json_rate", "Strict JSON"),
    ("schema_valid_rate", "Schema-valid"),
]


def load(results: Path, split: str) -> dict[str, dict[str, Any]]:
    out = {}
    for name in ROWS:
        p = results / name / f"{split}_metrics.json"
        if p.exists():
            out[name] = json.loads(p.read_text())
    return out


def table(metrics: dict[str, dict[str, Any]]) -> str:
    head = "| Model | " + " | ".join(c for _, c in COLS) + " |"
    sep = "|---" * (len(COLS) + 1) + "|"
    lines = [head, sep]
    for name, m in metrics.items():
        cells = [f"{100 * float(m[k]):.1f}" for k, _ in COLS]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def per_field_table(metrics: dict[str, dict[str, Any]]) -> str:
    names = list(metrics)
    lines = ["| Field | support | " + " | ".join(names) + " |", "|---" * (len(names) + 2) + "|"]
    any_m = next(iter(metrics.values()))
    for f in FIELDS:
        sup = any_m["per_field"][f]["support"]
        cells = [f"{100 * m['per_field'][f]['f1']:.1f}" for m in metrics.values()]
        lines.append(f"| {f} | {sup} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _micro_f1(counts: list[tuple[int, int, int]]) -> float:
    tp = sum(c[0] for c in counts)
    fp = sum(c[1] for c in counts)
    fn = sum(c[2] for c in counts)
    return 2 * tp / max(2 * tp + fp + fn, 1)


def paired_bootstrap(
    a: list[tuple[int, int, int]], b: list[tuple[int, int, int]], n_boot: int = 2000, seed: int = 0
) -> tuple[float, float, float]:
    """(delta, lo, hi) for micro-F1(b) - micro-F1(a), resampling queries."""
    rng = random.Random(seed)
    n = len(a)
    deltas = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        deltas.append(_micro_f1([b[i] for i in idx]) - _micro_f1([a[i] for i in idx]))
    deltas.sort()
    return (
        _micro_f1(b) - _micro_f1(a),
        deltas[int(0.025 * n_boot)],
        deltas[int(0.975 * n_boot)],
    )


def per_query_counts(results: Path, name: str, split: str) -> dict[str, tuple[int, int, int]]:
    out = {}
    for r in read_jsonl(results / name / f"{split}_predictions.jsonl"):
        c = count_pairs(r["pred"], r["gold"])
        out[r["id"]] = (c.tp, c.fp, c.fn)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=Path("results"))
    ap.add_argument("--split", default="test")
    args = ap.parse_args()
    metrics = load(args.results, args.split)
    if not metrics:
        raise SystemExit("no results yet")
    parts = ["## Results", "", table(metrics), "", "### Per-field F1", "", per_field_table(metrics)]
    pairs = [("base", "sft"), ("sft", "dpo"), ("sft", "grpo"), ("dpo", "sft_dpo_grpo")]
    ci_lines = []
    for a, b in pairs:
        if a in metrics and b in metrics:
            ca = per_query_counts(args.results, a, args.split)
            cb = per_query_counts(args.results, b, args.split)
            ids = sorted(set(ca) & set(cb))
            d, lo, hi = paired_bootstrap([ca[i] for i in ids], [cb[i] for i in ids])
            ci_lines.append(f"| {a} -> {b} | {100 * d:+.1f} | [{100 * lo:+.1f}, {100 * hi:+.1f}] |")
    if ci_lines:
        parts += ["", "### Paired bootstrap, micro-F1 delta (95% CI)", "",
                  "| Comparison | Delta | 95% CI |", "|---|---|---|", *ci_lines]  # fmt: skip
    text = "\n".join(parts) + "\n"
    (args.results / "summary.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
