import math

import pytest
import torch

from winnow.selection import select_channels, select_experts


def test_channel_selection_uses_ceil_and_stable_ties():
    scores = torch.ones(1, 4, 4)
    plan = select_channels(scores, 0.3, top_k=1)
    assert sum(plan.layers[0].widths) == math.ceil(0.3 * 16)
    assert plan.layers[0].experts == (0, 1)
    assert plan.layers[0].channels == ((0, 1, 2, 3), (0,))


def test_channel_selection_aligns_widths():
    scores = torch.arange(64, dtype=torch.float32).reshape(1, 4, 16)
    plan = select_channels(scores, 0.5, top_k=1, block_size=4)
    assert all(width % 4 == 0 for width in plan.layers[0].widths)
    assert sum(plan.layers[0].widths) == 32


def test_reap_selection_uses_ceil_and_original_order():
    scores = torch.tensor([[1.0, 4.0, 3.0, 2.0]])
    plan = select_experts(scores, 0.5, top_k=2, original_width=8)
    assert plan.layers[0].experts == (1, 2)
    assert plan.layers[0].widths == (8, 8)


def test_reap_rejects_fewer_experts_than_top_k():
    with pytest.raises(ValueError, match="router needs"):
        select_experts(torch.ones(1, 8), 0.1, top_k=2, original_width=8)
