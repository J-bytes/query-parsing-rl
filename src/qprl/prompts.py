"""Prompt construction shared by every stage.

The model sees the same system prompt in SFT, DPO, GRPO and evaluation, so a
change here changes all stages together. The prompt lists the schema but gives
no examples: the SFT stage teaches the format, and keeping the prompt short
keeps sequences short (the completions are ~20-60 tokens).
"""

from __future__ import annotations

from typing import Any, Protocol

from qprl.schema import FIELDS


class ChatTokenizer(Protocol):
    def apply_chat_template(self, conversation: Any, **kwargs: Any) -> Any: ...


def _schema_lines() -> str:
    return "\n".join(f"- {name}: {desc}" for name, (_, desc) in FIELDS.items())


SYSTEM_PROMPT = (
    "You parse e-commerce search queries into structured attributes.\n"
    "Return ONLY a JSON object, with no other text. Include a field only if the query "
    "states it; omit every other field. Copy wording from the query where possible "
    "(lowercase). Prices are numbers.\n"
    "Fields:\n" + _schema_lines()
)


def build_messages(query: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": query},
    ]


def render_prompt(tokenizer: ChatTokenizer, query: str) -> str:
    """Chat-templated prompt text ending with the assistant turn header."""
    text = tokenizer.apply_chat_template(
        build_messages(query), tokenize=False, add_generation_prompt=True
    )
    assert isinstance(text, str)
    return text
