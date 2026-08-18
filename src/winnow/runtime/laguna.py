"""Hugging Face reference runtime for pruned Laguna checkpoints."""

from __future__ import annotations

import torch
from torch import nn
from transformers.models.laguna.configuration_laguna import LagunaConfig
from transformers.models.laguna.modeling_laguna import LagunaForCausalLM

from .ragged import RaggedExperts


class WinnowLagunaConfig(LagunaConfig):
    """Add a variable-width expert table to the Laguna configuration.

    ``expert_widths`` and ``expert_indices`` carry one row per *sparse* decoder
    layer, in layer order; dense layers (``mlp_layer_types[i] == "dense"``) have
    no row.
    """

    model_type = "winnow_laguna"

    expert_widths: list[list[int]] | None = None
    expert_indices: list[list[int]] | None = None


class WinnowLagunaForCausalLM(LagunaForCausalLM):
    """Laguna with variable-width routed experts."""

    config_class = WinnowLagunaConfig

    def __init__(self, config: WinnowLagunaConfig) -> None:
        super().__init__(config)
        widths_table = config.expert_widths
        if widths_table is None:
            return
        moe_blocks = [
            layer.mlp for layer in self.model.layers if hasattr(layer.mlp, "experts")
        ]
        if len(widths_table) != len(moe_blocks):
            raise ValueError("expert_widths must contain one row for each sparse decoder layer")
        for block, widths in zip(moe_blocks, widths_table, strict=True):
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
            block.gate.e_score_correction_bias = nn.Parameter(
                torch.empty(
                    len(widths),
                    dtype=block.gate.e_score_correction_bias.dtype,
                    device=block.gate.e_score_correction_bias.device,
                ),
                requires_grad=False,
            )
            block.experts = RaggedExperts(config.hidden_size, list(widths), config.hidden_act)


__all__ = ["WinnowLagunaConfig", "WinnowLagunaForCausalLM"]
