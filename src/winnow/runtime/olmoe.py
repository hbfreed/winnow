"""Hugging Face reference runtime for pruned OLMoE checkpoints."""

from __future__ import annotations

from transformers.models.olmoe.configuration_olmoe import OlmoeConfig
from transformers.models.olmoe.modeling_olmoe import OlmoeForCausalLM

from .ragged import install_ragged_experts


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
            install_ragged_experts(layer.mlp, list(widths), config.hidden_size, config.hidden_act)


__all__ = ["WinnowOlmoeConfig", "WinnowOlmoeForCausalLM"]
