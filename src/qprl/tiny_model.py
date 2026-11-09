"""Build a tiny, randomly initialised Qwen2-architecture model for CPU tests.

Uses the real Qwen2.5 tokenizer and chat template (a ~10 MB download, cached),
so tokenisation, prompt rendering and EOS handling are exercised exactly as in
the real runs; only the transformer is shrunk to ~10M parameters (mostly the
151k-token embedding). No pretrained weights are downloaded.

    uv run --extra cpu python -m qprl.tiny_model --out .cache/tiny-qwen
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

TOKENIZER_SOURCE = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_DIR = Path(".cache/tiny-qwen")


def build_tiny_model(out: Path = DEFAULT_DIR, seed: int = 0) -> Path:
    from transformers import AutoTokenizer, Qwen2Config, Qwen2ForCausalLM

    if (out / "config.json").exists():
        return out
    tok = AutoTokenizer.from_pretrained(TOKENIZER_SOURCE)
    torch.manual_seed(seed)
    cfg = Qwen2Config(
        vocab_size=len(tok),
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=1024,
        tie_word_embeddings=True,
        attention_dropout=0.0,
        bos_token_id=tok.bos_token_id,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id,
    )
    model = Qwen2ForCausalLM(cfg)
    model.generation_config.eos_token_id = [tok.eos_token_id]
    model.generation_config.pad_token_id = tok.pad_token_id
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out)
    tok.save_pretrained(out)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=DEFAULT_DIR)
    args = ap.parse_args()
    path = build_tiny_model(args.out)
    print(path)


if __name__ == "__main__":
    main()
