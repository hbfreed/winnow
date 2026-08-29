"""Fused CUDA runtime for aligned Winnow experts."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from megablocks.backend.fused_moe import FusedMoEPlan, fused_moe_forward
from torch import nn


class FastRaggedMoE(nn.Module):
    """Pack variable-width experts for the fused MegaBlocks serving kernel."""

    block_size = 128

    def __init__(
        self,
        hidden_size: int,
        widths: list[int],
        top_k: int,
        *,
        dtype: torch.dtype | None = None,
        block_size: int | None = None,
    ) -> None:
        super().__init__()
        if block_size is not None:
            if block_size < 16 or block_size & (block_size - 1):
                raise ValueError(f"block_size must be a power of two >= 16, got {block_size}")
            self.block_size = block_size
        if any(width <= 0 or width % self.block_size for width in widths):
            raise ValueError(f"expert widths must be positive multiples of {self.block_size}")
        if top_k > len(widths):
            raise ValueError("top_k cannot be greater than the surviving expert count")
        self.hidden_size = int(hidden_size)
        self.expert_widths = tuple(int(width) for width in widths)
        self.num_experts = len(widths)
        self.top_k = int(top_k)
        self.total_width = sum(self.expert_widths)
        self.gate = nn.Linear(hidden_size, self.num_experts, bias=False, dtype=dtype)
        self.w_gate = nn.Parameter(torch.empty(hidden_size, self.total_width, dtype=dtype))
        self.w_up = nn.Parameter(torch.empty(hidden_size, self.total_width, dtype=dtype))
        self.w_down = nn.Parameter(torch.empty(self.total_width, hidden_size, dtype=dtype))
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

    @property
    def is_int8(self) -> bool:
        return self.w_gate.dtype == torch.int8

    def quantize_int8_(self) -> None:
        """Convert the packed expert weights to symmetric per-channel INT8.

        Gate/up scales follow their packed output columns ``[total_width]``;
        down scales are per expert and output channel ``[num_experts, hidden]``,
        matching :func:`winnow.runtime.int8_moe.fused_moe_forward_int8`.  The
        BF16 parameters are replaced by INT8 buffers, so a quantized module
        cannot be converted back.
        """
        if self.is_int8:
            return

        def _per_column(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            scale = weight.abs().amax(dim=0).float().clamp_min(1e-8) / 127.0
            quantized = torch.round(weight.float() / scale).clamp_(-127, 127).to(torch.int8)
            return quantized, scale

        gate, gate_scale = _per_column(self.w_gate.data)
        up, up_scale = _per_column(self.w_up.data)
        down = torch.empty_like(self.w_down.data, dtype=torch.int8)
        down_scale = torch.empty(
            self.num_experts, self.hidden_size, dtype=torch.float32, device=down.device
        )
        offset = 0
        for expert, width in enumerate(self.expert_widths):
            rows = self.w_down.data[offset : offset + width]
            scale = rows.abs().amax(dim=0).float().clamp_min(1e-8) / 127.0
            down[offset : offset + width] = (
                torch.round(rows.float() / scale).clamp_(-127, 127).to(torch.int8)
            )
            down_scale[expert] = scale
            offset += width

        for name in ("w_gate", "w_up", "w_down"):
            del self._parameters[name]
        self.register_buffer("w_gate", gate)
        self.register_buffer("w_up", up)
        self.register_buffer("w_down", down)
        self.register_buffer("w_gate_scale", gate_scale)
        self.register_buffer("w_up_scale", up_scale)
        self.register_buffer("w_down_scale", down_scale)

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
            if self.is_int8:
                from .int8_moe import fused_moe_forward_int8

                return fused_moe_forward_int8(
                    x,
                    weights.flatten(),
                    experts.flatten().int(),
                    self._plan(x.device),
                    self.top_k,
                    self.w_gate,
                    self.w_up,
                    self.w_down,
                    self.w_gate_scale,
                    self.w_up_scale,
                    self.w_down_scale,
                )
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


class FastSigmoidMoE(FastRaggedMoE):
    """Fused sigmoid-routed block with a shared expert (Laguna, Afmoe).

    The router selects on ``sigmoid(logits) + e_score_correction_bias`` but
    weights by the unbiased sigmoid scores, normalized over the top-k and
    multiplied by ``routed_scaling_factor``; the shared expert is added
    unweighted.  Router logit softcapping is not supported (both released
    Laguna 2.1 checkpoints ship with softcapping disabled).
    """

    def __init__(
        self,
        hidden_size: int,
        widths: list[int],
        top_k: int,
        routed_scaling_factor: float = 1.0,
        *,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__(hidden_size, widths, top_k, dtype=dtype)
        self.routed_scaling_factor = float(routed_scaling_factor)
        self.e_score_correction_bias = nn.Parameter(torch.zeros(len(widths)), requires_grad=False)
        self.shared_experts: nn.Module | None = None

    @torch.compiler.disable
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.shared_experts is None:
            raise RuntimeError("the shared expert is not attached")
        shape = hidden_states.shape
        x = hidden_states.reshape(-1, shape[-1])
        scores = torch.sigmoid(self.gate(x).float())
        _, experts = torch.topk(scores + self.e_score_correction_bias.float(), self.top_k, dim=-1)
        weights = scores.gather(-1, experts)
        weights = weights / weights.sum(dim=-1, keepdim=True)
        if self.routed_scaling_factor != 1.0:
            weights = weights * self.routed_scaling_factor
        routed = self._routed(x, weights.to(x.dtype), experts)
        return (routed + self.shared_experts(x)).reshape(shape)


__all__ = ["FastOlmoeMoE", "FastQwenMoE", "FastRaggedMoE", "FastSigmoidMoE"]
