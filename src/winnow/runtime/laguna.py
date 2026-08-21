"""Hugging Face reference runtime for pruned Laguna checkpoints."""

from __future__ import annotations

from transformers.models.laguna.configuration_laguna import LagunaConfig
from transformers.models.laguna.modeling_laguna import LagunaForCausalLM

from .ragged import install_ragged_experts


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
        moe_blocks = [layer.mlp for layer in self.model.layers if hasattr(layer.mlp, "experts")]
        if len(widths_table) != len(moe_blocks):
            raise ValueError("expert_widths must contain one row for each sparse decoder layer")
        for block, widths in zip(moe_blocks, widths_table, strict=True):
            install_ragged_experts(block, list(widths), config.hidden_size, config.hidden_act)


__all__ = ["WinnowLagunaConfig", "WinnowLagunaForCausalLM"]
