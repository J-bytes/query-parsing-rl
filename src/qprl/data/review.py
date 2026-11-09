"""Hand-review the test set, one query at a time, in the terminal.

    uv run --extra cpu python -m qprl.data.review data/processed/test.jsonl

For each unreviewed row it shows the query, the merged label and the rule
label, then asks:
    [enter] accept      e  edit as JSON      d  drop field(s)      s  skip
    x  exclude row (query is junk / not a shopping query)      q  save and quit
Progress is saved after every row, so it is safe to stop at any time. Edited
values are normalised with the same code the metric uses.

`--export-csv` / `--import-csv` round-trip the file through a spreadsheet for
anyone who prefers that to the terminal.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from qprl.schema import FIELDS, normalize_parse, to_target_json
from qprl.utils import read_jsonl, write_jsonl


def summary(rows: list[dict[str, Any]]) -> str:
    reviewed = sum(bool(r.get("reviewed")) for r in rows)
    excluded = sum(bool(r.get("excluded")) for r in rows)
    edited = sum(bool(r.get("edited")) for r in rows)
    return f"{reviewed}/{len(rows)} reviewed, {edited} edited, {excluded} excluded"


def apply_action(row: dict[str, Any], action: str, payload: str = "") -> bool:
    """Apply one review action. Returns False if the payload was invalid."""
    if action == "accept":
        row["reviewed"] = True
    elif action == "exclude":
        row["reviewed"] = True
        row["excluded"] = True
    elif action == "edit":
        try:
            obj = json.loads(payload)
        except json.JSONDecodeError:
            return False
        if not isinstance(obj, dict):
            return False
        row["label"] = normalize_parse(obj)
        row["reviewed"] = row["edited"] = True
    elif action == "drop":
        fields = [f.strip() for f in payload.replace(",", " ").split()]
        if not all(f in FIELDS for f in fields):
            return False
        row["label"] = {k: v for k, v in row["label"].items() if k not in fields}
        row["reviewed"] = row["edited"] = True
    else:
        raise ValueError(action)
    return True


def interactive(path: Path) -> None:
    rows = read_jsonl(path)
    print(summary(rows))
    for i, row in enumerate(rows):
        if row.get("reviewed"):
            continue
        while True:
            print(f"\n[{i + 1}/{len(rows)}]  {row['query']!r}")
            print(f"  label: {to_target_json(row['label'])}")
            if row.get("rule_label"):
                print(f"  rules: {to_target_json(row['rule_label'])}")
            cmd = input("  [enter]=accept e=edit d=drop s=skip x=exclude q=quit > ").strip().lower()
            if cmd == "":
                apply_action(row, "accept")
            elif cmd == "e":
                txt = input("  new JSON: ")
                if not apply_action(row, "edit", txt):
                    print("  invalid JSON object, try again")
                    continue
            elif cmd == "d":
                txt = input(f"  fields to drop ({', '.join(row['label'])}): ")
                if not apply_action(row, "drop", txt):
                    print("  unknown field name, try again")
                    continue
            elif cmd == "x":
                apply_action(row, "exclude")
            elif cmd == "s":
                break
            elif cmd == "q":
                write_jsonl(path, rows)
                print(summary(rows))
                return
            else:
                continue
            write_jsonl(path, rows)
            break
    write_jsonl(path, rows)
    print(summary(rows))


def export_csv(path: Path, out: Path) -> None:
    rows = read_jsonl(path)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "query", "label_json", "rule_label_json", "reviewed", "excluded"])
        for r in rows:
            w.writerow([
                r["id"], r["query"], to_target_json(r["label"]),
                to_target_json(r.get("rule_label", {})),
                int(bool(r.get("reviewed"))), int(bool(r.get("excluded"))),
            ])  # fmt: skip


def import_csv(path: Path, csv_path: Path) -> None:
    rows = {r["id"]: r for r in read_jsonl(path)}
    with open(csv_path, encoding="utf-8") as f:
        for rec in csv.DictReader(f):
            row = rows[rec["id"]]
            new_label = normalize_parse(json.loads(rec["label_json"]))
            row["edited"] = bool(row.get("edited")) or new_label != row["label"]
            row["label"] = new_label
            row["reviewed"] = rec["reviewed"] == "1"
            row["excluded"] = rec["excluded"] == "1"
    write_jsonl(path, rows.values())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("path", type=Path, nargs="?", default=Path("data/processed/test.jsonl"))
    ap.add_argument("--export-csv", type=Path)
    ap.add_argument("--import-csv", type=Path)
    ap.add_argument("--status", action="store_true")
    args = ap.parse_args()
    if args.status:
        print(summary(read_jsonl(args.path)))
    elif args.export_csv:
        export_csv(args.path, args.export_csv)
    elif args.import_csv:
        import_csv(args.path, args.import_csv)
    else:
        interactive(args.path)


if __name__ == "__main__":
    main()
