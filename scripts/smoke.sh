#!/usr/bin/env bash
# End-to-end smoke run of every stage on CPU with the tiny random model and the
# 24 synthetic fixture queries. Checks plumbing, not learning. ~1-2 minutes.
set -euo pipefail
TORCH="${TORCH:-cpu}"
PY="uv run --extra ${TORCH} python"
rm -rf runs/smoke
$PY -m qprl.sft.train  --config configs/smoke/sft.yaml
$PY -m qprl.dpo.sample --config configs/smoke/dpo.yaml
$PY -m qprl.dpo.train  --config configs/smoke/dpo.yaml
$PY -m qprl.grpo.train --config configs/smoke/grpo.yaml
$PY -m qprl.eval.run_eval --model runs/smoke/grpo/last --name smoke_grpo \
    --data tests/fixtures/queries.jsonl --results-dir runs/smoke/results --dtype fp32 --max-new-tokens 16
$PY -m qprl.eval.run_eval --rules --name rules --data tests/fixtures/queries.jsonl \
    --results-dir runs/smoke/results
for f in runs/smoke/sft/csv/metrics.csv runs/smoke/dpo/csv/metrics.csv runs/smoke/grpo/csv/metrics.csv \
         runs/smoke/grpo/rollouts.jsonl runs/smoke/results/smoke_grpo/queries_metrics.json; do
  test -s "$f" || { echo "SMOKE FAIL: missing $f"; exit 1; }
done
echo "SMOKE OK"
