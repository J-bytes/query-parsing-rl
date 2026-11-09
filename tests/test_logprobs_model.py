"""Tiny-model tests for log-prob scoring and generation plumbing."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from qprl.generation import generate
from qprl.logprobs import completion_token_logprobs, gather_logprobs
from qprl.tokenization import collate_prompt_completion

pytestmark = pytest.mark.slow


def log_softmax_reference(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return F.log_softmax(logits.float(), dim=-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)


def test_gather_matches_log_softmax() -> None:
    g = torch.Generator().manual_seed(0)
    logits = torch.randn(2, 5, 50, generator=g)
    tgt = torch.randint(0, 50, (2, 5), generator=g)
    assert torch.allclose(
        gather_logprobs(logits, tgt), log_softmax_reference(logits, tgt), atol=1e-6
    )


def test_padding_invariance(tiny_model_path: str) -> None:
    """A row's completion log-probs do not depend on how much left padding the
    batch forces on it: scoring alone == scoring next to a longer prompt."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tiny_model_path)
    model = AutoModelForCausalLM.from_pretrained(tiny_model_path).eval()
    short_p, long_p = [11, 12, 13], [21, 22, 23, 24, 25, 26, 27]
    comp = [31, 32, 33, 34]
    alone = collate_prompt_completion([short_p], [comp], tok.pad_token_id)
    both = collate_prompt_completion([short_p, long_p], [comp, comp[:2]], tok.pad_token_id)
    with torch.no_grad():
        a = completion_token_logprobs(model, alone["input_ids"], alone["attention_mask"], 4)
        b = completion_token_logprobs(model, both["input_ids"], both["attention_mask"], 4)
    assert torch.allclose(a[0], b[0], atol=1e-5)


def test_generation_groups_and_masks(tiny_model_path: str) -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tiny_model_path)
    model = AutoModelForCausalLM.from_pretrained(tiny_model_path)
    torch.manual_seed(0)
    out = generate(
        model, tok, ["a", "bbb ccc"], max_new_tokens=6, do_sample=True, num_return_sequences=3
    )
    assert out.completion_ids.shape[0] == 6 and len(out.texts) == 6
    assert torch.equal(out.prompt_ids[0], out.prompt_ids[2])  # rows grouped by prompt
    assert not torch.equal(out.prompt_ids[2], out.prompt_ids[3])
    # everything after the mask is padding
    assert bool((out.completion_ids[out.completion_mask == 0] == tok.pad_token_id).all())
