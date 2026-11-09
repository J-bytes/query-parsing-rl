from __future__ import annotations

from collections import Counter
from pathlib import Path

import pandas as pd

from qprl.data.build import apply_guard, merge_one
from qprl.data.esci import keep_query, sample_splits
from qprl.data.llm_label import build_label_messages, cache_key
from qprl.data.review import apply_action, export_csv, import_csv, summary
from qprl.utils import read_jsonl, write_jsonl


def test_keep_query() -> None:
    assert keep_query("red shoes", 12)
    assert not keep_query("", 12)
    assert not keep_query("a " * 13, 12)
    assert not keep_query("カメラ", 12)


def test_sample_splits_disjoint_and_respects_esci_split() -> None:
    df = pd.DataFrame({
        "query": [f"q{i}" for i in range(40)],
        "query_id": list(range(40)),
        "split": ["train"] * 30 + ["test"] * 10,
    })  # fmt: skip
    out = sample_splits(df, {"sft": 10, "dpo": 5, "grpo": 5, "dev": 4, "test": 6}, seed=0)
    ids = {k: {r["id"] for r in v} for k, v in out.items()}
    allids = [i for v in ids.values() for i in v]
    assert len(allids) == len(set(allids)) == 30
    test_qids = {f"esci-{i}" for i in range(30, 40)}
    assert ids["dev"] | ids["test"] <= test_qids
    assert not (ids["sft"] | ids["dpo"] | ids["grpo"]) & test_qids
    # test sample does not depend on dev size
    out2 = sample_splits(df, {"sft": 10, "dpo": 5, "grpo": 5, "dev": 1, "test": 6}, seed=0)
    assert out2["test"] == out["test"]


def test_guard_drops_hallucinated_values() -> None:
    stats: Counter[str] = Counter()
    lab = apply_guard(
        "Levi's mens jeans 32x30",
        {"brand": "levis", "color": "blue", "size": "32x30", "product_type": "denim pants",
         "gender": "men"},
        stats,
    )  # fmt: skip
    assert lab == {"brand": "levis", "size": "32x30", "gender": "men"}
    assert stats["guard_dropped_color"] == 1 and stats["guard_dropped_product_type"] == 1


def test_merge_price_override_and_invalid() -> None:
    stats: Counter[str] = Counter()
    lab = merge_one("lamp under $40", '{"product_type": "lamp", "price_max": 4}', stats)
    assert lab == {"product_type": "lamp", "price_max": 40.0}
    assert stats["rule_price_override"] == 1
    assert merge_one("lamp", "not json", stats) is None
    assert stats["llm_invalid_json"] == 1


def test_label_messages_and_cache_key() -> None:
    msgs = build_label_messages("red shoes", {"color": "red"})
    assert msgs[0]["role"] == "system" and "rule hints" in msgs[-1]["content"]
    assert cache_key("a", "m1") != cache_key("a", "m2")


def test_review_actions(tmp_path: Path) -> None:
    row = {"id": "x", "query": "red shoes", "label": {"color": "red", "brand": "zz"}}
    assert apply_action(row, "drop", "brand")
    assert row["label"] == {"color": "red"} and row["reviewed"] and row["edited"]
    assert not apply_action(row, "drop", "nonsense")
    assert not apply_action(row, "edit", "{bad")
    assert apply_action(row, "edit", '{"product_type": "Shoes", "color": "red"}')
    assert row["label"] == {"product_type": "shoes", "color": "red"}
    other = {"id": "y", "query": "q", "label": {}}
    apply_action(other, "exclude")
    assert other["excluded"]
    path = tmp_path / "t.jsonl"
    write_jsonl(path, [row, other, {"id": "z", "query": "q2", "label": {"brand": "a"}}])
    assert summary(read_jsonl(path)).startswith("2/3 reviewed")
    csv_path = tmp_path / "t.csv"
    export_csv(path, csv_path)
    import_csv(path, csv_path)
    assert read_jsonl(path)[0]["label"] == row["label"]
