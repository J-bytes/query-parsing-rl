# Weekend plan

Run order for one 24 GB GPU (RTX 3090 / 4090 class) with Qwen2.5-0.5B-Instruct,
full-parameter training in every stage.

**All runtimes below are estimates, not measurements.** They come from token
counts measured on the real data (prompt ≈ 260 tokens with the system prompt,
completion ≈ 35 tokens, max 96) and typical 0.5B throughput. Replace them with
the real numbers from `runs/*/csv/metrics.csv` once you have them; the README
should only ever quote measured times.

## Friday evening (≈ 1 h of your time, then let it run overnight)

| Step | Command | Estimate | Notes |
|---|---|---|---|
| Environment | `make setup-gpu` | 5 min | `uv sync --extra gpu`; PyPI torch wheels bundle CUDA |
| Sanity | `make test TORCH=gpu` | 2 min | 102 tests, CPU-only paths, includes TRL cross-checks |
| Data | `make data TORCH=gpu` | 1 min | ESCI examples parquet (49 MB). Add `make data-products` (1.06 GB) for the brand lexicon |
| Labelling | `make label TORCH=gpu LABELLER_BASE_URL=http://<unsloth-host>:<port>/v1 LABELLER_MODEL=<name>` | 20 min – 3 h | 20,800 queries. Depends entirely on the server: vLLM batches well, llama.cpp-style servers less so. Resumable: re-run the same command after any crash. Add `ARGS=--no-json-mode` if the server rejects `response_format` |

First check the endpoint with a 20-query run: `make label ARGS="--limit 20"`,
then look at `data/llm_cache/labels.jsonl`.

## Saturday

| Step | Command | Estimate | Notes |
|---|---|---|---|
| Build splits | `make build TORCH=gpu LABELLER_MODEL=<name>` | 1 min | Read `data/processed/build_stats.json`: how many labels the extractive guard dropped |
| Review test set | `make review TORCH=gpu` | 40–60 min (human) | 300 queries, resumable, `q` to save and quit. Do this before looking at any model output on test |
| Baselines | `make eval-rules eval-base TORCH=gpu` | 5 min | Rule labeller and zero-shot instruct model |
| SFT | `make sft TORCH=gpu` | 15–30 min | 2 epochs × 12k queries, effective batch 64, dev eval every 50 steps |
| Eval SFT | `make eval-sft TORCH=gpu` | 2 min | |
| DPO pairs | `make dpo-sample TORCH=gpu` | 10–20 min | 4,000 queries × k=8 samples. Check `dpo_pairs.stats.json`: if `all_tied` is most of the pool, raise temperature or k |
| DPO | `make dpo TORCH=gpu` | 10–20 min | Reference log-probs precomputed once, so no second model in memory |
| Eval DPO | `make eval-dpo TORCH=gpu` | 2 min | |
| GRPO cross-check on the real model | `uv run --extra gpu python -m qprl.dpo.trl_check --model runs/sft/best --pairs data/processed/dpo_pairs.jsonl --n 16` | 5 min | Optional: the CPU check already passed; this repeats it at real scale |

Start GRPO before going to bed (next row) if SFT and DPO look sane.

| Step | Command | Estimate | Notes |
|---|---|---|---|
| GRPO | `make grpo TORCH=gpu` | 1–1.5 h | 300 rollout batches × 128 completions, μ=2. HF `generate` dominates step time |

What to watch in TensorBoard (`uv run --extra gpu tensorboard --logdir runs`):
* `train/zero_std_fraction`: share of groups where all 8 samples tie. Near 1.0 means no learning signal. If SFT is already very good, most groups tie at the top; raise temperature to 1.2 or switch the pool to harder queries.
* `train/kl`: should grow slowly. A spike means the policy is drifting; lower lr.
* `train/clip_high`, `train/clip_low`: non-zero only because μ=2. A few percent is normal.
* `train/completion_len`: a rising length with flat reward is a reward-hacking smell.
* `dev/micro_f1`: the curve that matters.

## Sunday

| Step | Command | Estimate | Notes |
|---|---|---|---|
| Eval GRPO | `make eval-grpo TORCH=gpu` | 2 min | |
| Results | `make results TORCH=gpu` | 1 min | Table, per-field F1, paired bootstrap CIs, curve PNGs |
| Judge | `make judge TORCH=gpu` | 5–15 min | Label-free check of GRPO predictions |
| Failure analysis | read `results/*/test_predictions.jsonl` sorted by `f1` | 1 h | Fill the README failure-case section with real examples |
| README | fill the placeholders | 1 h | Only measured numbers |

### If time is left (pick one)

1. `make grpo-from-dpo`: the full SFT → DPO → GRPO chain (≈ 1–1.5 h).
2. Ablation `make grpo ARGS="loss_aggregation=grpo output_dir=runs/grpo_seqmean"`: length bias of per-sequence averaging vs DAPO token averaging.
3. Ablation `ARGS="advantage_scale=none"` (Dr. GRPO advantages).
4. `make gsm8k && make grpo-gsm8k`: second benchmark, comparable to published numbers (≈ 1.5–2 h; longer completions).

## Memory budget (estimate, 0.5B, bf16-mixed)

| Item | Size |
|---|---|
| fp32 master weights | 2.0 GB |
| fp32 gradients | 2.0 GB |
| AdamW states (2 × fp32) | 4.0 GB |
| bf16 reference copy (GRPO, β > 0) | 1.0 GB |
| Activations, one micro-batch of 16 × ~360 tokens, gradient checkpointing | 2–4 GB |
| Logits for log-probs (16 × 97 × 151,936 × 4 B) | ≈ 0.9 GB |

Comfortably under 24 GB. The 1.5B model needs `configs/sft_1p5b.yaml`
(bf16 weights + 8-bit AdamW); GRPO at 1.5B would also need a smaller
micro-batch. Not part of the weekend plan.
