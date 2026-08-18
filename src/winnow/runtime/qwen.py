"""Hugging Face reference runtime for pruned Qwen3.5 MoE checkpoints."""

from __future__ import annotations

from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import (
    Qwen3_5MoeTextConfig,
)
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    Qwen3_5MoeForCausalLM,
)

from .ragged import install_ragged_experts


class WinnowQwen3_5MoeConfig(Qwen3_5MoeTextConfig):
    """Add a variable-width expert table to the Qwen text configuration."""

    model_type = "winnow_qwen3_5_moe"

    def __init__(
        self,
        expert_widths: list[list[int]] | None = None,
        expert_indices: list[list[int]] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.expert_widths = expert_widths
        self.expert_indices = expert_indices


class WinnowQwen3_5MoeForCausalLM(Qwen3_5MoeForCausalLM):
    """Qwen3.5 MoE with variable-width routed experts."""

    config_class = WinnowQwen3_5MoeConfig

    def __init__(self, config: WinnowQwen3_5MoeConfig) -> None:
        super().__init__(config)
        widths_table = config.expert_widths
        if widths_table is None:
            return
        if len(widths_table) != len(self.model.layers):
            raise ValueError("expert_widths must contain one row for each decoder layer")
        for layer, widths in zip(self.model.layers, widths_table, strict=True):
            install_ragged_experts(layer.mlp, list(widths), config.hidden_size, config.hidden_act)


__all__ = ["WinnowQwen3_5MoeConfig", "WinnowQwen3_5MoeForCausalLM"]
