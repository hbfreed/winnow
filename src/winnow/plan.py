"""Pruning plan data."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class LayerPlan:
    """The original expert and channel indices that one layer keeps."""

    layer: int
    experts: tuple[int, ...]
    channels: tuple[tuple[int, ...], ...]

    @property
    def widths(self) -> tuple[int, ...]:
        return tuple(len(indices) for indices in self.channels)

    def to_dict(self) -> dict[str, Any]:
        return {
            "layer": self.layer,
            "experts": list(self.experts),
            "channels": [list(indices) for indices in self.channels],
            "widths": list(self.widths),
        }


@dataclass(frozen=True)
class PruningPlan:
    """A deterministic plan for all routed MoE layers."""

    strategy: str
    keep: float
    original_experts: int
    original_width: int
    top_k: int
    layers: tuple[LayerPlan, ...]
    block_size: int = 1

    @property
    def realized_channel_keep(self) -> float:
        kept = sum(sum(layer.widths) for layer in self.layers)
        total = len(self.layers) * self.original_experts * self.original_width
        return kept / total

    @property
    def realized_expert_keep(self) -> float:
        kept = sum(len(layer.experts) for layer in self.layers)
        total = len(self.layers) * self.original_experts
        return kept / total

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "keep": self.keep,
            "block_size": self.block_size,
            "original_experts": self.original_experts,
            "original_width": self.original_width,
            "top_k": self.top_k,
            "realized_channel_keep": self.realized_channel_keep,
            "realized_expert_keep": self.realized_expert_keep,
            "layers": [layer.to_dict() for layer in self.layers],
        }
