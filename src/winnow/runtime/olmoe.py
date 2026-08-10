"""Hugging Face reference runtime for pruned OLMoE checkpoints."""

from __future__ import annotations

import torch
from torch import nn
from transformers.models.olmoe.configuration_olmoe import OlmoeConfig
from transformers.models.olmoe.modeling_olmoe import OlmoeForCausalLM

from .ragged import RaggedExperts


class WinnowOlmoeConfig(OlmoeConfig):
    """Add a variable-width expert table to the OLMoE configuration."""

    model_type = "winnow_olmoe"

    def __init__(
        self,
        expert_widths: list[list[int]] | None = None,
        expert_indices: list[list[int]] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.expert_widths = expert_widths
        self.expert_indices = expert_indices


class WinnowOlmoeForCausalLM(OlmoeForCausalLM):
    """OLMoE with a variable count and width of routed experts."""

    config_class = WinnowOlmoeConfig

    def __init__(self, config: WinnowOlmoeConfig) -> None:
        super().__init__(config)
        widths_table = config.expert_widths
        if widths_table is None:
            return
        if len(widths_table) != len(self.model.layers):
            raise ValueError("expert_widths must contain one row for each decoder layer")
        for layer, widths in zip(self.model.layers, widths_table, strict=True):
            block = layer.mlp
            if len(widths) < block.gate.top_k:
                raise ValueError("each layer must keep at least the router top-k experts")
            if any(width <= 0 for width in widths):
                raise ValueError("expert widths must be greater than 0")
            old_weight = block.gate.weight
            block.gate.num_experts = len(widths)
            block.gate.weight = nn.Parameter(
                torch.empty(
                    len(widths),
                    config.hidden_size,
                    dtype=old_weight.dtype,
                    device=old_weight.device,
                )
            )
            block.experts = RaggedExperts(config.hidden_size, list(widths), config.hidden_act)


__all__ = ["WinnowOlmoeConfig", "WinnowOlmoeForCausalLM"]
