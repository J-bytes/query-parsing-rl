# One command per stage. Every target runs through uv.
#
#   make setup            CPU environment (this laptop, tests, smoke runs)
#   make setup-gpu        CUDA environment (the 24GB box)
#   make all TORCH=gpu    full pipeline after data is labelled
#
# TORCH selects the torch extra (cpu | gpu). Always go through these targets or
# `uv run --extra $(TORCH)`: a bare `uv run` re-syncs without the torch extra.

TORCH ?= cpu
UV    := uv run --extra $(TORCH)
PY    := $(UV) python
ARGS  ?=

# Labeller endpoint (OpenAI-compatible: vLLM, Unsloth/llama.cpp server, Ollama...)
LABELLER_BASE_URL ?=
LABELLER_MODEL    ?=
export LABELLER_BASE_URL LABELLER_MODEL

.PHONY: help setup setup-gpu tiny-model data data-products label build review review-status \
        gsm8k sft dpo-sample dpo grpo grpo-from-dpo grpo-gsm8k \
        eval-rules eval-base eval-sft eval-dpo eval-grpo eval-all results judge \
        all format lint typecheck deadcode test test-fast smoke trl-check mutate check index clean-smoke

help:  ## list targets
	@grep -E '^[a-zA-Z0-9_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  %-16s %s\n",$$1,$$2}'

# ------------------------------------------------------------------ setup ---
setup:  ## CPU env (tests, smoke)
	uv sync --extra cpu

setup-gpu:  ## CUDA env
	uv sync --extra gpu

tiny-model:  ## build the ~10M-param random Qwen2 used by tests (tokenizer download only)
	$(PY) -m qprl.tiny_model

# ------------------------------------------------------------------- data ---
data:  ## download ESCI queries (49 MB) and sample disjoint splits
	$(PY) -m qprl.data.esci --download

data-products:  ## also download ESCI products (1.06 GB) for the brand lexicon
	$(PY) -m qprl.data.esci --download --with-products

label:  ## LLM labelling pass over all splits (needs LABELLER_BASE_URL, LABELLER_MODEL)
	$(PY) -m qprl.data.llm_label --queries data/interim/queries_test.jsonl \
	  data/interim/queries_dev.jsonl data/interim/queries_sft.jsonl \
	  data/interim/queries_dpo.jsonl data/interim/queries_grpo.jsonl $(ARGS)

build:  ## merge rule + LLM labels into data/processed/*.jsonl
	$(PY) -m qprl.data.build --model "$(LABELLER_MODEL)"

review:  ## hand-review the 300-query test set (resumable)
	$(PY) -m qprl.data.review data/processed/test.jsonl

review-status:
	$(PY) -m qprl.data.review data/processed/test.jsonl --status

gsm8k:  ## optional: GSM8K splits (3 MB)
	$(PY) -m qprl.data.gsm8k

# --------------------------------------------------------------- training ---
sft:  ## stage 1
	$(PY) -m qprl.sft.train --config configs/sft.yaml $(ARGS)

dpo-sample:  ## stage 2a: sample k per query from SFT, build pairs
	$(PY) -m qprl.dpo.sample --config configs/dpo.yaml $(ARGS)

dpo:  ## stage 2b
	$(PY) -m qprl.dpo.train --config configs/dpo.yaml $(ARGS)

grpo:  ## stage 3 (from SFT)
	$(PY) -m qprl.grpo.train --config configs/grpo.yaml $(ARGS)

grpo-from-dpo:  ## optional: SFT -> DPO -> GRPO
	$(PY) -m qprl.grpo.train --config configs/grpo_from_dpo.yaml $(ARGS)

grpo-gsm8k:  ## optional second benchmark
	$(PY) -m qprl.grpo.train --config configs/grpo_gsm8k.yaml $(ARGS)

# ------------------------------------------------------------------- eval ---
EVAL := $(PY) -m qprl.eval.run_eval --data data/processed/test.jsonl

eval-rules:
	$(EVAL) --rules --name rules
eval-base:
	$(EVAL) --model Qwen/Qwen2.5-0.5B-Instruct --name base
eval-sft:
	$(EVAL) --model runs/sft/best --name sft
eval-dpo:
	$(EVAL) --model runs/dpo/best --name dpo
eval-grpo:
	$(EVAL) --model runs/grpo/best --name grpo
eval-all: eval-rules eval-base eval-sft eval-dpo eval-grpo  ## evaluate every stage on test

results:  ## markdown results table + bootstrap CIs -> results/summary.md
	$(PY) -m qprl.eval.compare
	$(PY) scripts/plot_curves.py

judge:  ## optional LLM-as-a-judge on GRPO predictions
	$(PY) -m qprl.eval.judge results/grpo/test_predictions.jsonl

all: sft eval-sft dpo-sample dpo eval-dpo grpo eval-grpo eval-rules eval-base results  ## full pipeline

# ---------------------------------------------------------------- quality ---
format:
	$(UV) ruff format src tests scripts
	$(UV) ruff check --fix src tests scripts

lint:  ## ruff
	$(UV) ruff check src tests scripts
	$(UV) ruff format --check src tests scripts

typecheck:  ## mypy --strict
	$(UV) mypy

deadcode:  ## vulture
	$(UV) vulture

test-fast:  ## unit + property tests (no model)
	$(UV) pytest -m "not slow and not trl"

test:  ## all tests incl. tiny-model and TRL cross-checks (CPU, ~1-2 min)
	$(UV) pytest

smoke: tiny-model  ## run every stage end to end for a few steps on CPU
	bash scripts/smoke.sh

trl-check:  ## DPO + GRPO numerical cross-check against TRL on the tiny model
	$(PY) -m qprl.dpo.trl_check
	$(PY) -m qprl.grpo.trl_check

mutate:  ## mutation testing of loss / reward / metric code (slow)
	$(UV) mutmut run
	$(UV) mutmut results

check: lint typecheck deadcode test  ## everything CI would run

index:  ## (re)index the repo in codebase-memory (needs the codebase-memory MCP server)
	@echo "Run from Claude Code: index_repository on $(CURDIR)"

clean-smoke:
	rm -rf runs/smoke
