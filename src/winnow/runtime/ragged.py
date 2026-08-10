"""Reference execution for variable-width fused experts."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from transformers.activations import ACT2FN


class RaggedExperts(nn.Module):
    """Run each surviving expert with its own SwiGLU width."""

    def __init__(self, hidden_size: int, widths: list[int], hidden_act: str) -> None:
        super().__init__()
        self.num_experts = len(widths)
        self.hidden_dim = hidden_size
        self.widths = tuple(int(width) for width in widths)
        self.gate_up_projs = nn.ParameterList(
            nn.Parameter(torch.empty(2 * width, hidden_size)) for width in self.widths
        )
        self.down_projs = nn.ParameterList(
            nn.Parameter(torch.empty(hidden_size, width)) for width in self.widths
        )
        self.act_fn = ACT2FN[hidden_act]

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        output = torch.zeros_like(hidden_states)
        with torch.no_grad():
            mask = F.one_hot(top_k_index, num_classes=self.num_experts).permute(2, 1, 0)
            hit = (mask.sum(dim=(-1, -2)) > 0).nonzero(as_tuple=False).flatten()
        for expert_tensor in hit:
            expert = int(expert_tensor)
            slots, tokens = torch.where(mask[expert])
            current = hidden_states[tokens]
            gate, up = F.linear(current, self.gate_up_projs[expert]).chunk(2, dim=-1)
            current = self.act_fn(gate) * up
            current = F.linear(current, self.down_projs[expert])
            current = current * top_k_weights[tokens, slots, None]
            output.index_add_(0, tokens, current.to(output.dtype))
        return output
