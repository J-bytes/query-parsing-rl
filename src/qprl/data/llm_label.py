"""LLM labelling pass through any OpenAI-compatible chat endpoint.

Works with vLLM, llama.cpp / Unsloth servers, Ollama, or a hosted API, as long
as it serves POST {base_url}/chat/completions. Configure with env vars or flags:

    LABELLER_BASE_URL   e.g. http://192.168.1.50:8000/v1
    LABELLER_MODEL      model name the server expects
    LABELLER_API_KEY    optional (sent as a Bearer token)

Design choices:
  * The prompt is the same schema description the student model sees, plus
    few-shot examples and the rule labeller's hints. Hints raise recall on
    price / gender patterns without forcing the LLM to accept them.
  * Results are cached per query in a JSONL file, so the run is resumable and a
    re-run never pays twice. The cache key includes the prompt version.
  * Temperature 0. One request per query, bounded concurrency.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import httpx

from qprl.data.rules import RuleLabeller
from qprl.prompts import SYSTEM_PROMPT
from qprl.schema import parse_completion, to_target_json
from qprl.utils import read_jsonl

PROMPT_VERSION = "v1"

# Hand-written few-shot examples (not from ESCI). They show omission of absent
# fields, surface-form copying, and price as a number.
FEW_SHOT: list[tuple[str, dict[str, Any]]] = [
    ("nike running shoes mens size 11 white", {
        "product_type": "running shoes", "brand": "nike", "gender": "men", "size": "11", "color": "white",
    }),
    ("iphone 13 pro case clear", {
        "product_type": "case", "compatible_with": "iphone 13 pro", "color": "clear",
    }),
    ("queen size bed frame wood under $200", {
        "product_type": "bed frame", "size": "queen", "material": "wood", "price_max": 200,
    }),
    ("toddler girl rain boots", {
        "product_type": "rain boots", "gender": "women", "age_group": "baby",
    }),
    ("light strips", {"product_type": "light strips"}),
    ("stainless steel water bottle 32 oz", {
        "product_type": "water bottle", "material": "stainless steel", "size": "32 oz",
    }),
]  # fmt: skip

LABEL_INSTRUCTIONS = (
    "\n\nLabelling notes:\n"
    "- product_type: the head noun phrase, without brand, colour, size or audience words.\n"
    "- gender for 'girls'/'boys' is women/men; age_group for them is kids; 'toddler' and "
    "'baby' are age_group baby.\n"
    "- Only use a price if the query states a bound ('under $50', 'less than 20 dollars').\n"
    "- The rule hints may be wrong or incomplete; trust the query text."
)


def build_label_messages(query: str, hints: dict[str, Any]) -> list[dict[str, str]]:
    msgs = [{"role": "system", "content": SYSTEM_PROMPT + LABEL_INSTRUCTIONS}]
    for q, parse in FEW_SHOT:
        msgs.append({"role": "user", "content": q})
        msgs.append({"role": "assistant", "content": to_target_json(parse)})
    hint_txt = f"\n(rule hints: {json.dumps(hints)})" if hints else ""
    msgs.append({"role": "user", "content": query + hint_txt})
    return msgs


def cache_key(query: str, model: str) -> str:
    return hashlib.sha1(f"{PROMPT_VERSION}|{model}|{query}".encode()).hexdigest()


async def _label_one(
    client: httpx.AsyncClient,
    sem: asyncio.Semaphore,
    url: str,
    model: str,
    query: str,
    hints: dict[str, Any],
    json_mode: bool,
    retries: int = 3,
) -> str:
    body: dict[str, Any] = {
        "model": model,
        "messages": build_label_messages(query, hints),
        "temperature": 0.0,
        "max_tokens": 200,
    }
    if json_mode:
        body["response_format"] = {"type": "json_object"}
    async with sem:
        for attempt in range(retries):
            try:
                r = await client.post(url, json=body)
                r.raise_for_status()
                content = r.json()["choices"][0]["message"]["content"]
                return str(content)
            except (httpx.HTTPError, KeyError, IndexError):
                if attempt == retries - 1:
                    raise
                await asyncio.sleep(2**attempt)
    raise AssertionError("unreachable")


async def label_file(
    queries_path: Path,
    cache_path: Path,
    base_url: str,
    model: str,
    api_key: str | None,
    brands_path: Path | None,
    concurrency: int,
    json_mode: bool,
    limit: int | None,
) -> None:
    rows = read_jsonl(queries_path)[:limit] if limit else read_jsonl(queries_path)
    rules = RuleLabeller.from_file(brands_path)
    done: set[str] = set()
    if cache_path.exists():
        done = {r["key"] for r in read_jsonl(cache_path)}
    todo = [r for r in rows if cache_key(r["query"], model) not in done]
    print(f"{queries_path.name}: {len(rows)} queries, {len(todo)} to label")
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    url = base_url.rstrip("/") + "/chat/completions"
    sem = asyncio.Semaphore(concurrency)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    async with httpx.AsyncClient(headers=headers, timeout=120.0) as client:

        async def work(row: dict[str, Any]) -> dict[str, Any]:
            hints = rules.label(row["query"])
            raw = await _label_one(client, sem, url, model, row["query"], hints, json_mode)
            return {
                "key": cache_key(row["query"], model),
                "id": row["id"],
                "query": row["query"],
                "model": model,
                "prompt_version": PROMPT_VERSION,
                "raw": raw,
                "ok": parse_completion(raw).lenient_json,
            }

        with open(cache_path, "a", encoding="utf-8") as f:
            for n, coro in enumerate(asyncio.as_completed([work(r) for r in todo]), start=1):
                rec = await coro
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                if n % 500 == 0:
                    f.flush()
                    print(f"  {n}/{len(todo)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--queries", type=Path, nargs="+", required=True)
    ap.add_argument("--cache", type=Path, default=Path("data/llm_cache/labels.jsonl"))
    ap.add_argument("--base-url", default=os.environ.get("LABELLER_BASE_URL"))
    ap.add_argument("--model", default=os.environ.get("LABELLER_MODEL"))
    ap.add_argument("--brands", type=Path, default=Path("data/interim/brands.txt"))
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--no-json-mode", action="store_true", help="server lacks response_format")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    if not args.base_url or not args.model:
        raise SystemExit("set LABELLER_BASE_URL and LABELLER_MODEL (or --base-url / --model)")
    for q in args.queries:
        asyncio.run(
            label_file(
                q,
                args.cache,
                args.base_url,
                args.model,
                os.environ.get("LABELLER_API_KEY"),
                args.brands,
                args.concurrency,
                not args.no_json_mode,
                args.limit,
            )
        )


if __name__ == "__main__":
    main()
