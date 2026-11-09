# query-parsing-rl

SFT → DPO → GRPO on a small model (Qwen2.5-0.5B-Instruct, full-parameter
fine-tuning) for **e-commerce query parsing**: turn a shopping query into
structured attributes. GRPO is implemented from scratch in plain PyTorch and
checked numerically against TRL.

```
"adidas soccer cleats boys"  ->  {"product_type": "soccer cleats", "brand": "adidas",
                                  "gender": "men", "age_group": "kids"}
```

> **Status:** code, tests and data pipeline are done and verified on CPU. The
> GPU runs have not been done yet, so every result below is a placeholder
> (`TBD`). Numbers will be filled in from `results/summary.md` after the runs;
> none are estimated or made up.

## Why this task

Query understanding sits in front of every product search engine: a parse
like the one above drives filters, boosts and facet selection. It is a good RL
testbed because:

* **The reward is verifiable.** Field-level F1 against a label plus a format
  bonus. No learned reward model, so there is nothing for the policy to fool
  except the labels themselves.
* **Outputs are short** (~35 tokens), so GRPO rollouts are cheap enough for one
  24 GB GPU over a weekend.
* **It has real structure to get wrong**: hallucinated brands, product type
  confused with brand, gender vs age for "boys", price bounds vs product
  amounts ("$25 gift card").

All data is public: Amazon's Shopping Queries Dataset (ESCI), Apache-2.0.

## Pipeline

```
ESCI queries ──► rule labeller ─┐
                                ├─► merge + extractive guard ─► train/dev/test
LLM labeller (OpenAI-compatible)┘                                 │
                                                       test: 300 hand-reviewed
                                                                  │
Qwen2.5-0.5B-Instruct ──SFT──► π_sft ──sample k=8, score──► pairs ──DPO──► π_dpo
                                  │
                                  └──────────GRPO (from scratch)──────────► π_grpo
                                  
every stage ──► same eval harness (field P/R/F1, JSON validity, exact match)
```

| Stage | What it does | Code |
|---|---|---|
| Data | Sample disjoint splits from ESCI (train pools from ESCI-train, dev/test from ESCI-test). Rule labels + LLM labels, merged with an extractive guard. | `src/qprl/data/` |
| Eval | Field-level micro P/R/F1, per-field F1, strict / lenient JSON rate, schema-valid rate, exact match, paired bootstrap CIs, optional label-free LLM judge. | `src/qprl/eval/` |
| SFT | Lightning, completion-only cross-entropy, full parameters. | `src/qprl/sft/` |
| DPO | On-policy pairs from π_sft scored by the verifier; DPO loss written out (≈ 20 lines); reference log-probs precomputed once. | `src/qprl/dpo/` |
| GRPO | Group-relative advantages, clipped surrogate, k3 KL, four loss aggregations (GRPO / BNPO / DAPO / Dr. GRPO). Pure tensor functions + a Lightning manual-optimisation loop. | `src/qprl/grpo/` |

### Output schema

Flat JSON; a field is **omitted** when the query does not state it.

| Field | Type | Example |
|---|---|---|
| `product_type` | str | "running shoes" |
| `brand` | str | "nike" |
| `color` | str | "rose gold" |
| `size` | str | "xl", "queen", "55 inch", "32 oz" |
| `material` | str | "stainless steel" |
| `gender` | men / women / unisex | |
| `age_group` | baby / kids / teen / adult | |
| `price_min`, `price_max` | number | 50 |
| `compatible_with` | str | "iphone 12" |

Values are compared after normalisation (lowercase, punctuation stripped,
apostrophes deleted, prices rounded to cents). There is deliberately no
stemming: "shoe" ≠ "shoes". The labelling prompt and the extractive guard make
labels surface forms from the query, which keeps the strict metric fair.

## Design decisions (and why)

**Labels.** The rule labeller (regexes + a 5,265-brand lexicon built from ESCI
product data) is precise but low-recall; an LLM is high-recall but
hallucinates. The merge takes the LLM label, lets rule prices override when
they carry an explicit `$` or currency word (regexes beat a 7B model on "under
$35" vs "$35 gift card"), and drops any extractive value that does not appear
verbatim in the query. Drop counts are written to
`data/processed/build_stats.json`. The test set is then hand-reviewed with a
resumable terminal tool (`make review`).

**Reward** = `1.0 · F1(pred, gold) + 0.2 · 1[strict JSON and schema-valid]`.
F1 is over (field, value) pairs; a wrong value is one FP and one FN. Empty
prediction on an empty label scores F1 = 1; unparseable output scores 0. The
format bonus is small so the model cannot trade content for formatting.

**DPO pairs are on-policy.** Chosen / rejected are the best / worst of 8
samples from π_sft, with a minimum reward gap of 0.2. Pairs sampled from π_ref
have sensible likelihood under it, so the log-ratios in the DPO loss are well
scaled. Gold-as-chosen is available as an ablation (`sample.gold_chosen=true`).

**Reference model in DPO costs nothing at train time.** π_ref = π_sft = the
initial policy, so one no-grad pass before training stores
log π_ref(chosen) and log π_ref(rejected); the reference never sits in GPU
memory during training.

**GRPO defaults** (`configs/grpo.yaml`): G = 8 samples per prompt, 16 prompts
per step, μ = 2 optimisation passes per batch (so the importance ratio and
clipping actually do something), ε_low = 0.2 / ε_high = 0.28 ("clip-higher"
from DAPO), β = 0.02 k3 KL to π_sft, DAPO token-level aggregation (no length
bias), truncated completions masked out. The main health metric is
`train/zero_std_fraction`: groups whose 8 rewards tie carry no gradient.

**Full fine-tuning, no LoRA.** fp32 master weights + bf16 autocast + gradient
checkpointing. 0.5B fits easily (≈ 8 GB weights, grads and AdamW state); a
1.5B config with bf16 weights and 8-bit AdamW is included but not part of the
weekend plan.

**Positions under left padding.** Log-probs are computed with `position_ids`
derived from the attention mask, the same rule `generate` uses, so training
scores exactly what the sampler produced.

## Verification

Everything below runs on CPU (`make check`) with a ~5M-parameter random
Qwen2-architecture model that uses the real Qwen2.5 tokenizer and chat template.

### Cross-checks against TRL 1.17.1 (measured)

| Check | Method | Result |
|---|---|---|
| DPO end to end | Ours vs `DPOTrainer`, same init, full-batch, constant LR AdamW, 5 steps, fp32 | step-0 loss = log 2 in both; max per-step loss difference **6.6e-7** |
| GRPO advantages | Run TRL's `_generate_and_score_completions`, re-score its completions, feed to our `group_advantages` | max difference **1.2e-7** (group), **0** (batch, none) |
| GRPO loss + gradients | Same batch through TRL's `_compute_loss` and our `grpo_loss`, with perturbed π_old / π_ref so ~21% of tokens are clipped and KL ≠ 0 | for all 4 aggregations × KL-ratio-correction on/off: loss difference ≤ **3.7e-8**, max per-parameter gradient difference ≤ **9e-8** |

Reproduce: `make trl-check` (writes `results/trl_check_{dpo,grpo}.json`).

One finding from this: TRL 1.17 multiplies the KL term by the importance
ratio π_θ/π_old by default (`use_bias_correction_kl=True`). The original GRPO
paper does not. Our first comparison differed by 5e-4 until this was matched;
it is now an explicit option (`kl_ratio_correction`, default off, only matters
when μ > 1).

### Tests

`make test`: 102 tests, ~60 s on CPU.

* Unit tests with hand-computed values for the DPO loss, advantages, k3 KL
  (including a check that its expectation equals the exact KL), clipping in
  both directions, every loss aggregation, the length bias of per-sequence
  averaging, the reward, every metric, the rule labeller and the data merge.
* Property-based tests (Hypothesis): JSON round-trip, normalisation
  idempotence, F1 bounds and symmetry, zero-mean advantages per group, DPO shift
  invariance, `parse_completion` never raises.
* Learning tests: GRPO on a 5-armed bandit finds the best arm; DPO on a
  4-way categorical moves mass to the chosen item.
* Tiny-model tests: log-prob gather vs `log_softmax`, left-padding invariance,
  grouped generation.
* Smoke test: SFT → DPO sampling → DPO → GRPO → eval end to end for a few steps.

Quality gates: `ruff`, `mypy --strict` (src, tests, scripts), `vulture`, and
`mutmut` on the maths core (schema, reward, metrics, DPO loss, GRPO
algorithm). The first mutmut run killed 589 / 684 mutants; the survivors pointed
at untested paths (the lower clip bound, normaliser pass-through, garbage
output on an empty label, per-field FP counting), which now have tests. Current
run: 636 / 684 killed, 11 timeouts, 36 survivors, all reviewed as equivalent
mutants (dtype / device defaults, error-message text, `sum(+1)` on 2-D tensors,
`zip(strict=...)` after an explicit length check, float-equality boundaries).

## Results

Test set: 300 ESCI-test queries, hand-reviewed. Greedy decoding.

| Model | Field F1 | P | R | Exact match | Strict JSON | Schema-valid |
|---|---|---|---|---|---|---|
| Rules | TBD | TBD | TBD | TBD | TBD | TBD |
| Qwen2.5-0.5B-Instruct (zero-shot) | TBD | TBD | TBD | TBD | TBD | TBD |
| + SFT | TBD | TBD | TBD | TBD | TBD | TBD |
| + DPO | TBD | TBD | TBD | TBD | TBD | TBD |
| + GRPO (from SFT) | TBD | TBD | TBD | TBD | TBD | TBD |

Paired bootstrap 95% CIs on the F1 deltas, per-field F1 and the LLM-judge
numbers go here (`make results`, `make judge`).

### Training curves

`results/curves/{sft,dpo,grpo}.png` (from `make results`). TBD.

### Failure cases

TBD after the runs: read `results/*/test_predictions.jsonl` sorted by `f1`.
Categories worth checking, based on what the data looks like:

* **Price support is tiny.** On the 500 dev queries the rule labeller finds a
  price bound in one. Per-field price F1 will be noisy; report support.
* Brand vs product line ("sonicare essence replacement heads": brand is
  Philips, "sonicare" is a line).
* Part-number style queries ("3/8 in. od compression x 3/8 in. flare brass
  adapter fitting").
* Non-shopping or intent queries ("meat ebt eligible food").

## Setup

```bash
make setup          # CPU env: tests and smoke runs
make setup-gpu      # CUDA env on the GPU box
make test           # 102 tests incl. TRL cross-checks
make smoke          # every stage end to end on CPU
```

`TORCH=cpu|gpu` picks the torch build. Always go through `make` or
`uv run --extra $TORCH`: a bare `uv run` re-syncs without a torch extra.

Then follow [WEEKEND_PLAN.md](WEEKEND_PLAN.md). In short:

```bash
make data TORCH=gpu                          # ESCI queries, 49 MB
make label TORCH=gpu LABELLER_BASE_URL=... LABELLER_MODEL=...
make build review TORCH=gpu LABELLER_MODEL=...
make all TORCH=gpu                           # SFT, DPO, GRPO, eval, results
```

The labeller and judge talk to any OpenAI-compatible endpoint (vLLM, an
Unsloth / llama.cpp server, Ollama, or a hosted API).

## Repo layout

```
configs/            one YAML per stage; smoke/ variants for CPU
src/qprl/
  schema.py         fields, parsing (strict + lenient), normalisation
  reward.py         field F1 + format bonus
  prompts.py        system prompt shared by every stage
  logprobs.py       per-token log-probs of completion tokens
  generation.py     batched sampling, EOS masks
  tasks.py          query parsing + GSM8K behind one interface
  training.py       optimizer, schedule, loggers, dev-eval callback
  data/             ESCI sampling, rules, LLM labelling, merge, review, GSM8K
  eval/             metrics, eval CLI, results table + bootstrap, LLM judge
  sft/ dpo/ grpo/   the three stages; dpo/ and grpo/ also hold trl_check.py
tests/              unit, property, tiny-model, TRL and smoke tests
scripts/            smoke.sh, plot_curves.py
```

## With more compute

* **vLLM for rollouts.** HF `generate` dominates GRPO step time. Serving the
  policy in vLLM with weight sync each step would give larger groups and more
  steps for the same wall clock.
* **Larger models**: Qwen2.5-1.5B / 3B with ZeRO-2 or FSDP across GPUs instead
  of 8-bit optimisers.
* **Better labels**: a stronger labeller plus a second human-reviewed slice for
  dev, so model selection does not rely on LLM labels.
* **Harder prompt pool for GRPO**: keep queries where π_sft's group reward
  variance is high (curriculum by `zero_std_fraction`).
* **Reward shaping ablations**: per-field weights (brand errors cost more than
  colour errors in search), penalising unsupported fields more than missing ones.
* **Seeds**: three seeds per stage for honest error bars on top of the
  bootstrap CIs.

## Licence

Code: Apache-2.0. Data: ESCI (Apache-2.0); GSM8K (MIT).
