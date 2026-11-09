"""Optional LLM-as-a-judge check that does not use the gold labels.

Field F1 measures agreement with labels that were themselves produced by an
LLM and a human pass. A judge that sees only (query, predicted parse) gives an
independent signal: is every predicted field supported by the query, and was
anything obvious missed? It also catches reward hacking that the F1 reward
cannot see (e.g. a field that happens to match a wrong label).

    uv run --extra gpu python -m qprl.eval.judge results/grpo/test_predictions.jsonl

Uses the same OpenAI-compatible endpoint settings as the labeller
(LABELLER_BASE_URL / LABELLER_MODEL, or JUDGE_BASE_URL / JUDGE_MODEL).
Reports: share of predicted fields judged supported (precision-like), share of
queries with a judged omission (recall-like), and an overall 1-5 score.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
from typing import Any

import httpx

from qprl.schema import to_target_json
from qprl.utils import read_jsonl, write_jsonl

JUDGE_PROMPT = """You check a structured parse of an e-commerce search query.
Query: {query}
Parse: {parse}

For each field in the parse, decide whether the query supports it.
Then say whether the parse missed any attribute clearly stated in the query
(brand, colour, size, material, gender, age group, price bound, compatible device, product type).
Answer with JSON only:
{{"unsupported_fields": [<field names>], "missed": [<short descriptions>], "score": <1-5>}}"""


async def _judge_one(
    client: httpx.AsyncClient, sem: asyncio.Semaphore, url: str, model: str, row: dict[str, Any]
) -> dict[str, Any]:
    prompt = JUDGE_PROMPT.format(query=row["query"], parse=to_target_json(row["pred"]))
    body = {"model": model, "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0, "max_tokens": 200}  # fmt: skip
    async with sem:
        r = await client.post(url, json=body)
        r.raise_for_status()
        raw = str(r.json()["choices"][0]["message"]["content"])
    # Judge output is not a query parse, so read it with json directly.
    verdict: dict[str, Any]
    try:
        start = raw.find("{")
        verdict, _ = json.JSONDecoder().raw_decode(raw[start:]) if start >= 0 else ({}, 0)
    except json.JSONDecodeError:
        verdict = {}
    if not isinstance(verdict, dict):
        verdict = {}
    return {"id": row["id"], "query": row["query"], "pred": row["pred"], "verdict": verdict}


def summarise(judged: list[dict[str, Any]]) -> dict[str, float]:
    n_fields = sum(len(j["pred"]) for j in judged)
    unsupported = sum(
        len([f for f in j["verdict"].get("unsupported_fields", []) if f in j["pred"]])
        for j in judged
    )
    missed = sum(bool(j["verdict"].get("missed")) for j in judged)
    scores = [float(j["verdict"]["score"]) for j in judged if "score" in j["verdict"]]
    n = max(len(judged), 1)
    return {
        "judge_supported_field_rate": 1 - unsupported / max(n_fields, 1),
        "judge_query_with_omission_rate": missed / n,
        "judge_mean_score": sum(scores) / max(len(scores), 1),
        "judge_parse_failures": float(sum(not j["verdict"] for j in judged)),
    }


async def judge_file(path: Path, base_url: str, model: str, concurrency: int) -> dict[str, float]:
    rows = read_jsonl(path)
    url = base_url.rstrip("/") + "/chat/completions"
    key = os.environ.get("JUDGE_API_KEY") or os.environ.get("LABELLER_API_KEY")
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    sem = asyncio.Semaphore(concurrency)
    async with httpx.AsyncClient(headers=headers, timeout=120.0) as client:
        judged = await asyncio.gather(*[_judge_one(client, sem, url, model, r) for r in rows])
    out = path.with_name(path.stem.replace("_predictions", "") + "_judge.jsonl")
    write_jsonl(out, judged)
    summary = summarise(list(judged))
    path.with_name(path.stem.replace("_predictions", "") + "_judge_summary.json").write_text(
        json.dumps(summary, indent=2)
    )
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("predictions", type=Path)
    ap.add_argument(
        "--base-url",
        default=os.environ.get("JUDGE_BASE_URL") or os.environ.get("LABELLER_BASE_URL"),
    )
    ap.add_argument(
        "--model", default=os.environ.get("JUDGE_MODEL") or os.environ.get("LABELLER_MODEL")
    )
    ap.add_argument("--concurrency", type=int, default=16)
    args = ap.parse_args()
    if not args.base_url or not args.model:
        raise SystemExit("set JUDGE_BASE_URL/JUDGE_MODEL or LABELLER_BASE_URL/LABELLER_MODEL")
    asyncio.run(judge_file(args.predictions, args.base_url, args.model, args.concurrency))


if __name__ == "__main__":
    main()
