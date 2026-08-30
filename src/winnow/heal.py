"""LoRA-style healing for pruned Winnow checkpoints.

Healing trains three things on the reference ragged runtime and nothing else:
per-expert LoRA on the ragged expert matrices (where pruning removed mass),
LoRA on the attention projections, and the router gate in full (its score
distribution is what pruning distorts most).  ``merge_healing`` folds every
adapter back into the base weights, so the healed model saves as a standard
Winnow checkpoint.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from .adapters import block_router
from .runtime.ragged import RaggedExperts

_ATTENTION_PROJECTIONS = ("q_proj", "k_proj", "v_proj", "o_proj")


class LoraLinear(nn.Module):
    """A frozen linear with a trainable low-rank residual."""

    def __init__(self, base: nn.Linear, rank: int, alpha: float) -> None:
        super().__init__()
        self.base = base
        self.base.weight.requires_grad_(False)
        self.scale = alpha / rank
        dtype = base.weight.dtype
        device = base.weight.device
        self.lora_a = nn.Parameter(torch.empty(rank, base.in_features, dtype=dtype, device=device))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, rank, dtype=dtype, device=device))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + (x @ self.lora_a.t() @ self.lora_b.t()) * self.scale

    def merged_weight(self) -> torch.Tensor:
        return self.base.weight + (self.lora_b @ self.lora_a) * self.scale


class HealingRaggedExperts(RaggedExperts):
    """Ragged experts with a trainable low-rank residual per expert matrix."""

    def __init__(self, source: RaggedExperts, rank: int, alpha: float) -> None:
        super().__init__(source.hidden_dim, list(source.widths), "silu")
        self.act_fn = source.act_fn  # keep the source activation regardless of name
        self.scale = alpha / rank
        dtype = source.gate_up_projs[0].dtype
        with torch.no_grad():
            for slot in range(self.num_experts):
                self.gate_up_projs[slot] = source.gate_up_projs[slot]
                self.down_projs[slot] = source.down_projs[slot]
        for parameter in self.parameters():
            parameter.requires_grad_(False)

        def _pair(out_features: int, in_features: int, device) -> tuple[nn.Parameter, nn.Parameter]:
            a = nn.Parameter(torch.empty(rank, in_features, dtype=dtype, device=device))
            b = nn.Parameter(torch.zeros(out_features, rank, dtype=dtype, device=device))
            nn.init.kaiming_uniform_(a, a=math.sqrt(5))
            return a, b

        gate_up_pairs = [
            _pair(2 * width, self.hidden_dim, self.gate_up_projs[slot].device)
            for slot, width in enumerate(self.widths)
        ]
        down_pairs = [
            _pair(self.hidden_dim, width, self.down_projs[slot].device)
            for slot, width in enumerate(self.widths)
        ]
        self.lora_gate_up_a = nn.ParameterList(a for a, _b in gate_up_pairs)
        self.lora_gate_up_b = nn.ParameterList(b for _a, b in gate_up_pairs)
        self.lora_down_a = nn.ParameterList(a for a, _b in down_pairs)
        self.lora_down_b = nn.ParameterList(b for _a, b in down_pairs)

    def _effective(self, kind: str, expert: int) -> torch.Tensor:
        base = getattr(self, f"{kind}_projs")[expert]
        a = getattr(self, f"lora_{kind}_a")[expert]
        b = getattr(self, f"lora_{kind}_b")[expert]
        return base + (b @ a) * self.scale

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        from .runtime.ragged import routed_token_segments

        output = torch.zeros_like(hidden_states)
        _counts, segments = routed_token_segments(top_k_index, self.num_experts)
        for expert, tokens, slots in segments:
            current = hidden_states[tokens]
            gate, up = nn.functional.linear(current, self._effective("gate_up", expert)).chunk(
                2, dim=-1
            )
            current = self.act_fn(gate) * up
            current = nn.functional.linear(current, self._effective("down", expert))
            current = current * top_k_weights[tokens, slots, None]
            output.index_add_(0, tokens, current.to(output.dtype))
        return output

    @torch.no_grad()
    def merge_(self) -> RaggedExperts:
        merged = RaggedExperts(self.hidden_dim, list(self.widths), "silu")
        merged.act_fn = self.act_fn
        for slot in range(self.num_experts):
            merged.gate_up_projs[slot] = nn.Parameter(self._effective("gate_up", slot).detach())
            merged.down_projs[slot] = nn.Parameter(self._effective("down", slot).detach())
        return merged


def attach_healing(model: nn.Module, *, rank: int = 16, alpha: float = 32.0) -> list[str]:
    """Freeze the model and install the healing adapters.

    Returns the names of the trainable parameters: expert LoRA, attention
    LoRA, and the (fully trained) router gate weight.
    """
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for layer in model.model.layers:
        attention = layer.self_attn
        for name in _ATTENTION_PROJECTIONS:
            setattr(attention, name, LoraLinear(getattr(attention, name), rank, alpha))
        block = layer.mlp
        if hasattr(block, "experts"):
            block.experts = HealingRaggedExperts(block.experts, rank, alpha)
            router = block_router(block)
            gate = router if hasattr(router, "weight") else router.gate
            gate.weight.requires_grad_(True)
    return [name for name, parameter in model.named_parameters() if parameter.requires_grad]


def merge_healing(model: nn.Module) -> nn.Module:
    """Fold every adapter back into base weights, in place."""
    for layer in model.model.layers:
        attention = layer.self_attn
        for name in _ATTENTION_PROJECTIONS:
            module = getattr(attention, name)
            if isinstance(module, LoraLinear):
                with torch.no_grad():
                    module.base.weight.copy_(module.merged_weight())
                setattr(attention, name, module.base)
        block = layer.mlp
        if isinstance(getattr(block, "experts", None), HealingRaggedExperts):
            block.experts = block.experts.merge_()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


__all__ = ["HealingRaggedExperts", "LoraLinear", "attach_healing", "merge_healing"]
