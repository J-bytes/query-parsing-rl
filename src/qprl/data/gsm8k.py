"""Optional second benchmark: GSM8K (MIT licence), for comparability with
published GRPO results. Downloads ~3 MB from the Hugging Face Hub.

    uv run --extra gpu python -m qprl.data.gsm8k --out-dir data/processed/gsm8k
"""

from __future__ import annotations

import argparse
from pathlib import Path

from qprl.utils import write_jsonl


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=Path("data/processed/gsm8k"))
    ap.add_argument("--dev", type=int, default=200, help="held out from GSM8K train")
    args = ap.parse_args()
    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main")
    for split in ("train", "test"):
        rows = []
        for i, ex in enumerate(ds[split]):
            solution = ex["answer"]
            final = solution.split("####")[-1].strip().replace(",", "")
            rows.append(
                {"id": f"gsm8k-{split}-{i}", "question": ex["question"], "answer": final,
                 "solution": solution}
            )  # fmt: skip
        if split == "train":
            write_jsonl(args.out_dir / "dev.jsonl", rows[: args.dev])
            write_jsonl(args.out_dir / "grpo.jsonl", rows[args.dev :])
        else:
            write_jsonl(args.out_dir / "test.jsonl", rows)
        print(split, len(rows))


if __name__ == "__main__":
    main()
