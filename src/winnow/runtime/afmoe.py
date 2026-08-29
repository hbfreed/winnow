"""Hugging Face reference runtime for pruned Afmoe (Arcee Trinity) checkpoints."""

from __future__ import annotations

from transformers.models.afmoe.configuration_afmoe import AfmoeConfig
from transformers.models.afmoe.modeling_afmoe import AfmoeForCausalLM

from .ragged import install_ragged_experts


class WinnowAfmoeConfig(AfmoeConfig):
    """Add a variable-width expert table to the Afmoe configuration.

    ``expert_widths`` and ``expert_indices`` carry one row per *sparse* decoder
    layer, in layer order; the first ``num_dense_layers`` layers have no row.
    """

    model_type = "winnow_afmoe"

    expert_widths: list[list[int]] | None = None
    expert_indices: list[list[int]] | None = None


class WinnowAfmoeForCausalLM(AfmoeForCausalLM):
    """Afmoe with variable-width routed experts."""

    config_class = WinnowAfmoeConfig

    def __init__(self, config: WinnowAfmoeConfig) -> None:
        super().__init__(config)
        widths_table = config.expert_widths
        if widths_table is None:
            return
        moe_blocks = [layer.mlp for layer in self.model.layers if hasattr(layer.mlp, "experts")]
        if len(widths_table) != len(moe_blocks):
            raise ValueError("expert_widths must contain one row for each sparse decoder layer")
        for block, widths in zip(moe_blocks, widths_table, strict=True):
            install_ragged_experts(block, list(widths), config.hidden_size, config.hidden_act)


__all__ = ["WinnowAfmoeConfig", "WinnowAfmoeForCausalLM"]
