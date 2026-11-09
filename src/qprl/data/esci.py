"""Load and sample queries from Amazon's Shopping Queries (ESCI) dataset.

Dataset: https://github.com/amazon-science/esci-data (Apache-2.0).
Files used:
  shopping_queries_dataset_examples.parquet  (~49 MB): query, query_id, locale, split
  shopping_queries_dataset_products.parquet  (~1.06 GB, optional): product_brand,
      used only to build a brand lexicon for the rule labeller.

We use the US locale only and only the query text: no product relevance labels
are needed for query parsing. ESCI's own train/test split is respected: our
train pools come from ESCI train queries, our dev and test sets from ESCI test
queries, so no query string appears on both sides.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import urllib.request
from collections import Counter
from pathlib import Path

import pandas as pd

from qprl.schema import normalize_text
from qprl.utils import write_jsonl

BASE_URL = "https://github.com/amazon-science/esci-data/raw/main/shopping_queries_dataset/"
EXAMPLES = "shopping_queries_dataset_examples.parquet"
PRODUCTS = "shopping_queries_dataset_products.parquet"
# sha256 of the files as downloaded on 2026-11-08. A mismatch is a warning, not an
# error: upstream could re-publish, and the user should know if that happened.
SHA256 = {
    EXAMPLES: "4a735b693b4a424a6fc67f5be6e4c811495c488bbf66d02a602d308b2744263a",
    PRODUCTS: "25124442d064d64b26f74082d6fa09438d679efc0c183cf28d19064a2b65a265",
}

_ASCII_QUERY = re.compile(r"^[\x20-\x7e]+$")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(raw_dir: Path, with_products: bool) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    names = [EXAMPLES] + ([PRODUCTS] if with_products else [])
    for name in names:
        dest = raw_dir / name
        if not dest.exists():
            print(f"downloading {name} ...")
            urllib.request.urlretrieve(BASE_URL + name, dest)
        digest = sha256_file(dest)
        status = "ok" if digest == SHA256[name] else "MISMATCH (upstream changed?)"
        print(f"{name}: sha256 {digest[:16]}... {status}")


def keep_query(q: str, max_words: int) -> bool:
    """Short, printable-ASCII, non-empty queries. Long natural-language requests
    (some ESCI queries are full sentences) are out of scope for a parser."""
    words = q.split()
    return 1 <= len(words) <= max_words and bool(_ASCII_QUERY.match(q))


def load_us_queries(raw_dir: Path, max_words: int = 12) -> pd.DataFrame:
    df = pd.read_parquet(
        raw_dir / EXAMPLES, columns=["query", "query_id", "product_locale", "split"]
    )
    df = df[df["product_locale"] == "us"].drop_duplicates("query_id")
    df["query"] = df["query"].str.strip()
    df = df[df["query"].map(lambda q: keep_query(q, max_words))]
    # Same text can have several query_ids; dedupe on normalised text and keep
    # one split per text (test wins, so test strings never leak into train).
    df["norm"] = df["query"].map(normalize_text)
    df = df.sort_values("split", ascending=False)  # "train" > "test", so test rows come last
    df = df.drop_duplicates("norm", keep="last")
    return df[["query", "query_id", "split", "norm"]].reset_index(drop=True)


def build_brand_lexicon(raw_dir: Path, min_products: int, out_path: Path) -> int:
    """Brands that appear on at least `min_products` US products, normalised.

    Very short or very generic "brands" (e.g. 'generic', 'unknown') are dropped.
    This lexicon only *suggests* brands to the LLM labeller and drives the rule
    baseline; it never overrides the LLM label.
    """
    df = pd.read_parquet(raw_dir / PRODUCTS, columns=["product_locale", "product_brand"])
    df = df[df["product_locale"] == "us"]
    counts = Counter(normalize_text(b) for b in df["product_brand"].dropna())
    banned = {"generic", "unknown", "unbranded", "na", "n a", "none", "no brand", "other"}
    brands = sorted(
        b for b, c in counts.items() if c >= min_products and len(b) >= 3 and b not in banned
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(brands) + "\n", encoding="utf-8")
    return len(brands)


def sample_splits(
    df: pd.DataFrame, sizes: dict[str, int], seed: int
) -> dict[str, list[dict[str, str]]]:
    """Disjoint samples. sft/dpo/grpo pools from ESCI train; dev/test from ESCI test."""
    train = df[df["split"] == "train"].sample(frac=1.0, random_state=seed)
    test = df[df["split"] == "test"].sample(frac=1.0, random_state=seed)
    out: dict[str, list[dict[str, str]]] = {}
    i = 0
    for name in ("sft", "dpo", "grpo"):
        n = sizes[name]
        out[name] = [
            {"id": f"esci-{r.query_id}", "query": str(r.query)}
            for r in train.iloc[i : i + n].itertuples()
        ]
        i += n
    j = 0
    for name in ("test", "dev"):  # test first so its sample does not depend on dev size
        n = sizes[name]
        out[name] = [
            {"id": f"esci-{r.query_id}", "query": str(r.query)}
            for r in test.iloc[j : j + n].itertuples()
        ]
        j += n
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    ap.add_argument("--out-dir", type=Path, default=Path("data/interim"))
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--with-products", action="store_true")
    ap.add_argument("--min-brand-products", type=int, default=30)
    ap.add_argument("--max-words", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sft", type=int, default=12000)
    ap.add_argument("--dpo", type=int, default=4000)
    ap.add_argument("--grpo", type=int, default=4000)
    ap.add_argument("--dev", type=int, default=500)
    ap.add_argument("--test", type=int, default=300)
    args = ap.parse_args()
    if args.download:
        download(args.raw_dir, args.with_products)
    if (args.raw_dir / PRODUCTS).exists():
        n = build_brand_lexicon(args.raw_dir, args.min_brand_products, args.out_dir / "brands.txt")
        print(f"brand lexicon: {n} brands")
    df = load_us_queries(args.raw_dir, args.max_words)
    print(f"US queries after filtering: {len(df)} ({(df['split'] == 'test').sum()} ESCI-test)")
    sizes = {k: getattr(args, k) for k in ("sft", "dpo", "grpo", "dev", "test")}
    for name, rows in sample_splits(df, sizes, args.seed).items():
        write_jsonl(args.out_dir / f"queries_{name}.jsonl", rows)
        print(f"{name}: {len(rows)} queries")


if __name__ == "__main__":
    main()
