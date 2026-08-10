"""Fused CUDA runtime for aligned Winnow experts."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from megablocks.backend.fused_moe import FusedMoEPlan, fused_moe_forward
from torch import nn


class FastRaggedMoE(nn.Module):
    """Pack variable-width experts for the fused MegaBlocks serving kernel."""

    block_size = 128

    def __init__(self, hidden_size: int, widths: list[int], top_k: int) -> None:
        super().__init__()
        if any(width <= 0 or width % self.block_size for width in widths):
            raise ValueError("expert widths must be positive multiples of 128")
        if top_k > len(widths):
            raise ValueError("top_k cannot be greater than the surviving expert count")
        self.hidden_size = int(hidden_size)
        self.expert_widths = tuple(int(width) for width in widths)
        self.num_experts = len(widths)
        self.top_k = int(top_k)
        self.total_width = sum(self.expert_widths)
        self.gate = nn.Linear(hidden_size, self.num_experts, bias=False)
        self.w_gate = nn.Parameter(torch.empty(hidden_size, self.total_width))
        self.w_up = nn.Parameter(torch.empty(hidden_size, self.total_width))
        self.w_down = nn.Parameter(torch.empty(self.total_width, hidden_size))
        self._plan_cache: FusedMoEPlan | None = None

    def _offset(self, expert: int) -> int:
        return sum(self.expert_widths[:expert])

    def load_expert_weight_(
        self, expert: int, projection: str, checkpoint_weight: torch.Tensor
    ) -> str:
        """Load one checkpoint matrix into the packed fused layout."""
        if not 0 <= expert < self.num_experts:
            raise IndexError(f"expert index {expert} is out of range")
        if projection not in {"gate", "up", "down"}:
            raise ValueError(f"unknown projection {projection!r}")
        width = self.expert_widths[expert]
        expected = (self.hidden_size, width) if projection == "down" else (width, self.hidden_size)
        if tuple(checkpoint_weight.shape) != expected:
            raise ValueError(
                f"expert {expert} {projection} has shape {tuple(checkpoint_weight.shape)}; "
                f"expected {expected}"
            )
        offset = self._offset(expert)
        destination = (
            self.w_down.data[offset : offset + width]
            if projection == "down"
            else getattr(self, f"w_{projection}").data[:, offset : offset + width]
        )
        destination.copy_(checkpoint_weight.to(destination.device).t())
        return f"w_{projection}"

    def expert_weight(self, expert: int, projection: str) -> torch.Tensor:
        """Return one expert matrix in checkpoint layout."""
        width = self.expert_widths[expert]
        offset = self._offset(expert)
        if projection == "down":
            return self.w_down[offset : offset + width].t()
        if projection not in {"gate", "up"}:
            raise ValueError(f"unknown projection {projection!r}")
        return getattr(self, f"w_{projection}")[:, offset : offset + width].t()

    def _plan(self, device: torch.device) -> FusedMoEPlan:
        if self._plan_cache is None or self._plan_cache.expert_col_off.device != device:
            self._plan_cache = FusedMoEPlan(
                self.expert_widths,
                self.hidden_size,
                device,
                block_n=self.block_size,
            )
        return self._plan_cache

    def _routed(
        self, x: torch.Tensor, weights: torch.Tensor, experts: torch.Tensor
    ) -> torch.Tensor:
        if x.device.type != "cuda":
            raise RuntimeError("the fused Winnow runtime requires CUDA")
        if x.dtype not in {torch.bfloat16, torch.float16}:
            raise RuntimeError("the fused Winnow runtime requires BF16 or FP16")
        if torch.is_grad_enabled() and (
            x.requires_grad or any(parameter.requires_grad for parameter in self.parameters())
        ):
            raise RuntimeError("the fused Winnow runtime is inference-only")
        with torch.cuda.device(x.device):
            return fused_moe_forward(
                x,
                weights.flatten(),
                experts.flatten().int(),
                self._plan(x.device),
                self.top_k,
                self.w_gate,
                self.w_up,
                self.w_down,
            )

    def _route(self, x: torch.Tensor, *, normalize: bool) -> tuple[torch.Tensor, torch.Tensor]:
        probabilities = F.softmax(self.gate(x), dim=-1, dtype=torch.float32)
        weights, experts = torch.topk(probabilities, self.top_k, dim=-1)
        if normalize:
            weights /= weights.sum(dim=-1, keepdim=True)
        return weights.to(x.dtype), experts


class FastOlmoeMoE(FastRaggedMoE):
    """Fused OLMoE block with unnormalized selected router weights."""

    @torch.compiler.disable
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        shape = hidden_states.shape
        x = hidden_states.reshape(-1, shape[-1])
        weights, experts = self._route(x, normalize=False)
        return self._routed(x, weights, experts).reshape(shape)


class FastQwenMoE(FastRaggedMoE):
    """Fused Qwen block with normalized routing and an unchanged shared expert."""

    def __init__(self, hidden_size: int, widths: list[int], top_k: int) -> None:
        super().__init__(hidden_size, widths, top_k)
        self.shared_expert: nn.Module | None = None
        self.shared_expert_gate: nn.Module | None = None

    @torch.compiler.disable
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.shared_expert is None or self.shared_expert_gate is None:
            raise RuntimeError("the Qwen shared expert is not attached")
        shape = hidden_states.shape
        x = hidden_states.reshape(-1, shape[-1])
        weights, experts = self._route(x, normalize=True)
        shared = torch.sigmoid(self.shared_expert_gate(x)) * self.shared_expert(x)
        return (self._routed(x, weights, experts) + shared).reshape(shape)


__all__ = ["FastOlmoeMoE", "FastQwenMoE", "FastRaggedMoE"]
