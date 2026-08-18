"""INT8-W8A16 specialization of the ragged grouped MoE serving kernel.

The pinned ``megablocks-variable`` forward owns routing, row planning, and the
BF16 expert path.  This module reuses those routing primitives and replaces
only the two expert GEMMs when weights are stored as symmetric per-output-
channel INT8:

* gate/up scales are packed like their output columns: ``[sum(widths)]``;
* down scales are per expert and output channel: ``[num_experts, hidden]``.

Activations remain FP16/BF16.  Each INT8 weight tile is cast to the activation
dtype immediately before ``tl.dot`` and the FP32 accumulator is multiplied by
the output-channel scale afterwards.  This is the same W8A16 computation used
by vLLM's legacy ``experts_int8`` Triton path and works on Ampere (SM80+); it
does not require Marlin or native FP8 hardware.

Inference only.  Ported from the GLEAN project's ``int8_moe`` module.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from megablocks.backend import kernels
from megablocks.backend.fused_moe import (
    _count_experts,
    _plan_tiles,
    _scatter_routes,
)


@triton.jit
def _gate_up_silu_int8(
    x,
    wg,
    wu,
    wg_scale,
    wu_scale,
    h,
    route_tokens,
    tile_expert,
    expert_col_off,
    expert_nblocks,
    HIDDEN: tl.constexpr,
    TOTAL_W: tl.constexpr,
    MAX_W: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    COMPUTE_TYPE: tl.constexpr,
):
    """Fused gate/up/SwiGLU with per-output-channel INT8 weights."""
    mt = tl.program_id(0)
    nt = tl.program_id(1)
    e = tl.load(tile_expert + mt)
    if e < 0:
        return
    if nt >= tl.load(expert_nblocks + e):
        return

    rows = mt * BLOCK_M + tl.arange(0, BLOCK_M)
    tok = tl.load(route_tokens + rows)
    valid = tok >= 0
    xp = x + tl.maximum(tok, 0)[:, None] * HIDDEN

    local = nt * BLOCK_N + tl.arange(0, BLOCK_N)
    col = tl.load(expert_col_off + e) + local

    acc_g = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc_u = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, HIDDEN, BLOCK_K):
        kk = k + tl.arange(0, BLOCK_K)
        a = tl.load(xp + kk[None, :], mask=valid[:, None], other=0.0)
        off = kk[:, None] * TOTAL_W + col[None, :]
        # W8A16: the MMA remains FP16/BF16.  Only the resident weight is INT8.
        bg = tl.load(wg + off).to(COMPUTE_TYPE)
        bu = tl.load(wu + off).to(COMPUTE_TYPE)
        acc_g = tl.dot(a, bg, acc=acc_g)
        acc_u = tl.dot(a, bu, acc=acc_u)

    acc_g *= tl.load(wg_scale + col)[None, :]
    acc_u *= tl.load(wu_scale + col)[None, :]
    gate = acc_g * tl.sigmoid(acc_g)
    tl.store(
        h + rows[:, None] * MAX_W + local[None, :],
        (gate * acc_u).to(h.dtype.element_ty),
    )


@triton.jit
def _down_proj_int8(
    h,
    wd,
    wd_scale,
    y,
    tile_expert,
    expert_col_off,
    expert_nblocks,
    HIDDEN: tl.constexpr,
    MAX_W: tl.constexpr,
    WIDTH_BLOCK: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    COMPUTE_TYPE: tl.constexpr,
):
    """Down projection with per-(expert, output-channel) INT8 scales."""
    mt = tl.program_id(0)
    nt = tl.program_id(1)
    e = tl.load(tile_expert + mt)
    if e < 0:
        return

    width = tl.load(expert_nblocks + e) * WIDTH_BLOCK
    coff = tl.load(expert_col_off + e)
    rows = mt * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = nt * BLOCK_N + tl.arange(0, BLOCK_N)

    hp = h + rows[:, None] * MAX_W
    wdp = wd + cols[None, :]
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, width, BLOCK_K):
        kk = k + tl.arange(0, BLOCK_K)
        a = tl.load(hp + kk[None, :])
        b = tl.load(wdp + (coff + kk)[:, None] * HIDDEN).to(COMPUTE_TYPE)
        acc = tl.dot(a, b, acc=acc)

    acc *= tl.load(wd_scale + e * HIDDEN + cols)[None, :]
    tl.store(
        y + rows[:, None] * HIDDEN + cols[None, :],
        acc.to(y.dtype.element_ty),
    )


def _compute_type(dtype: torch.dtype):
    if dtype == torch.bfloat16:
        return tl.bfloat16
    if dtype == torch.float16:
        return tl.float16
    raise ValueError(f"INT8 W8A16 requires fp16/bf16 activations, got {dtype}")


def _validate_weights_and_scales(
    plan,
    hidden: int,
    w_gate: torch.Tensor,
    w_up: torch.Tensor,
    w_down: torch.Tensor,
    w_gate_scale: torch.Tensor,
    w_up_scale: torch.Tensor,
    w_down_scale: torch.Tensor,
) -> None:
    expected_gate = (hidden, plan.total_width)
    expected_down = (plan.total_width, hidden)
    if w_gate.dtype != torch.int8 or w_up.dtype != torch.int8:
        raise TypeError("INT8 W8A16 gate/up weights must have dtype torch.int8")
    if w_down.dtype != torch.int8:
        raise TypeError("INT8 W8A16 down weights must have dtype torch.int8")
    if tuple(w_gate.shape) != expected_gate or tuple(w_up.shape) != expected_gate:
        raise ValueError(
            f"gate/up weights must have shape {expected_gate}, got "
            f"{tuple(w_gate.shape)} and {tuple(w_up.shape)}"
        )
    if tuple(w_down.shape) != expected_down:
        raise ValueError(
            f"down weight must have shape {expected_down}, got {tuple(w_down.shape)}"
        )
    if tuple(w_gate_scale.shape) != (plan.total_width,):
        raise ValueError("gate scale must have one value per packed output channel")
    if tuple(w_up_scale.shape) != (plan.total_width,):
        raise ValueError("up scale must have one value per packed output channel")
    if tuple(w_down_scale.shape) != (plan.num_experts, hidden):
        raise ValueError("down scale must have shape [num_experts, hidden_size]")
    if any(
        scale.dtype != torch.float32
        for scale in (w_gate_scale, w_up_scale, w_down_scale)
    ):
        raise TypeError("INT8 W8A16 scales must have dtype torch.float32")
    tensors = (
        w_gate,
        w_up,
        w_down,
        w_gate_scale,
        w_up_scale,
        w_down_scale,
    )
    if any(t.device != w_gate.device for t in tensors):
        raise ValueError("all INT8 weights and scales must be on one device")


def fused_moe_forward_int8(
    x: torch.Tensor,
    top_weights: torch.Tensor,
    expert_ids: torch.Tensor,
    plan,
    top_k: int,
    w_gate: torch.Tensor,
    w_up: torch.Tensor,
    w_down: torch.Tensor,
    w_gate_scale: torch.Tensor,
    w_up_scale: torch.Tensor,
    w_down_scale: torch.Tensor,
    num_warps: int | None = None,
    num_stages: int | None = None,
) -> torch.Tensor:
    """Run the ragged grouped MoE with INT8 weights and FP16/BF16 acts."""
    num_tokens, hidden = x.shape
    num_routes = num_tokens * top_k
    if hidden != plan.hidden_size:
        raise ValueError(f"expected hidden {plan.hidden_size}, got {hidden}")
    if expert_ids.numel() != num_routes:
        raise ValueError(
            f"expected {num_routes} expert ids, got {expert_ids.numel()}"
        )
    if hidden % plan.block_n:
        raise ValueError(
            f"hidden size {hidden} must be divisible by block {plan.block_n}"
        )
    _validate_weights_and_scales(
        plan,
        hidden,
        w_gate,
        w_up,
        w_down,
        w_gate_scale,
        w_up_scale,
        w_down_scale,
    )
    compute_type = _compute_type(x.dtype)

    bn = plan.block_n
    bm, warps, stages = plan.tile_config(num_tokens, top_k)
    num_warps = warps if num_warps is None else num_warps
    num_stages = stages if num_stages is None else num_stages
    m_tiles, row_bound = plan.bounds(num_tokens, top_k, bm)
    dev = x.device

    tile_expert = torch.empty(m_tiles, dtype=torch.int32, device=dev)
    route_tokens = torch.empty(row_bound, dtype=torch.int32, device=dev)
    route_rows = torch.empty(num_routes, dtype=torch.int32, device=dev)

    plan.counts.zero_()
    _count_experts[(triton.cdiv(num_routes, 1024),)](
        expert_ids,
        plan.counts,
        num_routes,
        NUM_EXPERTS=plan.num_experts,
        BINS=plan.count_bins,
        BLOCK=1024,
        num_warps=4,
    )
    _plan_tiles[(1,)](
        plan.counts,
        plan.expert_row_start,
        plan.cursor,
        tile_expert,
        route_tokens,
        m_tiles,
        NUM_EXPERTS=plan.num_experts,
        BLOCK_E=plan.block_e,
        BLOCK_M=bm,
        FILL=1024,
        num_warps=4,
    )
    _scatter_routes[(triton.cdiv(num_routes, 256),)](
        expert_ids,
        route_tokens,
        route_rows,
        plan.cursor,
        plan.expert_row_start,
        num_routes,
        TOP_K=top_k,
        BLOCK_X=256,
        num_warps=4,
    )

    h = torch.empty((row_bound, plan.max_width), dtype=x.dtype, device=dev)
    _gate_up_silu_int8[(m_tiles, plan.max_nblocks)](
        x,
        w_gate,
        w_up,
        w_gate_scale,
        w_up_scale,
        h,
        route_tokens,
        tile_expert,
        plan.expert_col_off,
        plan.expert_nblocks,
        HIDDEN=hidden,
        TOTAL_W=plan.total_width,
        MAX_W=plan.max_width,
        BLOCK_M=bm,
        BLOCK_N=bn,
        BLOCK_K=64,
        COMPUTE_TYPE=compute_type,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    y = torch.empty((row_bound, hidden), dtype=x.dtype, device=dev)
    _down_proj_int8[(m_tiles, hidden // bn)](
        h,
        w_down,
        w_down_scale,
        y,
        tile_expert,
        plan.expert_col_off,
        plan.expert_nblocks,
        HIDDEN=hidden,
        MAX_W=plan.max_width,
        WIDTH_BLOCK=bn,
        BLOCK_M=bm,
        BLOCK_N=bn,
        BLOCK_K=64,
        COMPUTE_TYPE=compute_type,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    out = torch.empty((num_tokens, hidden), dtype=x.dtype, device=dev)
    kernels._scatter_reduce[(num_tokens,)](
        out,
        y,
        route_rows,
        top_weights,
        NUM_COLUMNS=hidden,
        TOP_K=top_k,
        SCALE=top_weights is not None,
    )
    return out
