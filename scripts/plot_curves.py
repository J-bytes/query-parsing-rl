"""Plot training curves from the Lightning CSV logs into results/curves/*.png.

make results
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

PANELS = {
    "sft": ["train/loss", "dev/micro_f1", "dev/strict_json_rate"],
    "dpo": ["train/loss", "train/reward_accuracy", "train/reward_margin", "dev/micro_f1"],
    "grpo": [
        "train/reward_mean",
        "train/zero_std_fraction",
        "train/kl",
        "train/clip_high",
        "train/completion_len",
        "dev/micro_f1",
    ],
}


def plot_run(name: str, metrics_csv: Path, out_dir: Path) -> Path | None:
    if not metrics_csv.exists():
        return None
    df = pd.read_csv(metrics_csv)
    cols = [c for c in PANELS[name] if c in df.columns]
    if not cols:
        return None
    fig, axes = plt.subplots(1, len(cols), figsize=(4 * len(cols), 3.2), squeeze=False)
    for ax, col in zip(axes[0], cols, strict=True):
        s = df[["step", col]].dropna()
        ax.plot(s["step"], s[col], marker="o" if col.startswith("dev/") else None, ms=3)
        ax.set_title(col)
        ax.set_xlabel("optimizer step")
        ax.grid(alpha=0.3)
    fig.suptitle(name.upper())
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def main() -> None:
    for name in PANELS:
        p = plot_run(name, Path("runs") / name / "csv" / "metrics.csv", Path("results/curves"))
        if p:
            print(p)


if __name__ == "__main__":
    main()
