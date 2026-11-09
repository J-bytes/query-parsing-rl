from __future__ import annotations

import torch

from qprl.generation import completion_mask_from_ids
from qprl.tokenization import collate_prompt_completion, common_prefix_length, pad_left, pad_right


def test_common_prefix() -> None:
    assert common_prefix_length([1, 2, 3], [1, 2, 4]) == 2
    assert common_prefix_length([], [1]) == 0
    assert common_prefix_length([1, 2], [1, 2, 3]) == 2


def test_padding_layout() -> None:
    ids, mask = pad_left([[1, 2], [3]], 0)
    assert ids.tolist() == [[1, 2], [0, 3]] and mask.tolist() == [[1, 1], [0, 1]]
    ids, mask = pad_right([[1, 2], [3]], 0)
    assert ids.tolist() == [[1, 2], [3, 0]] and mask.tolist() == [[1, 1], [1, 0]]
    b = collate_prompt_completion([[1, 2], [3]], [[7], [8, 9]], 0)
    assert b["input_ids"].tolist() == [[1, 2, 7, 0], [0, 3, 8, 9]]
    assert b["completion_mask"].tolist() == [[1, 0], [1, 1]]


def test_completion_mask_first_eos() -> None:
    ids = torch.tensor([[5, 9, 6, 9], [5, 6, 7, 8], [9, 1, 1, 1]])
    m = completion_mask_from_ids(ids, [9])
    assert m.tolist() == [[1, 1, 0, 0], [1, 1, 1, 1], [1, 0, 0, 0]]
    m2 = completion_mask_from_ids(ids, [9, 7])
    assert m2[1].tolist() == [1, 1, 1, 0]
