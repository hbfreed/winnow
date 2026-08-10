"""Deterministic channel and expert selection."""

from __future__ import annotations

import math

import torch

from .plan import LayerPlan, PruningPlan
from .retention import validate_keep


def _aligned_widths(widths: torch.Tensor, block_size: int, full_width: int) -> torch.Tensor:
    if block_size <= 0:
        raise ValueError("block_size must be greater than 0")
    if full_width % block_size:
        raise ValueError("block_size must divide the original expert width")
    widths = widths.to(torch.long)
    if block_size == 1:
        return widths

    trimmed = widths.clone()
    trimmed[trimmed < block_size] = 0
    aligned = (trimmed // block_size) * block_size
    block_budget = int(widths.sum()) // block_size
    remaining = block_budget - int(aligned.sum()) // block_size
    survivors = (trimmed > 0).nonzero(as_tuple=False).flatten()
    if remaining <= 0 or survivors.numel() == 0:
        return aligned

    remainders = trimmed[survivors] - aligned[survivors]
    order = survivors[torch.argsort(remainders, descending=True, stable=True)].tolist()
    while remaining > 0:
        changed = False
        for expert in order:
            if remaining == 0:
                break
            if int(aligned[expert]) + block_size > full_width:
                continue
            aligned[expert] += block_size
            remaining -= 1
            changed = True
        if not changed:
            break
    return aligned


def select_channels(
    scores: torch.Tensor,
    keep: float,
    *,
    top_k: int,
    block_size: int = 1,
    layer_indices: tuple[int, ...] | None = None,
) -> PruningPlan:
    """Select channels by a stable per-layer global rank."""
    keep = validate_keep(keep)
    if scores.ndim != 3:
        raise ValueError("channel scores must have shape [layers, experts, channels]")
    if not torch.isfinite(scores).all():
        raise ValueError("channel scores must be finite")
    layers, experts, width = map(int, scores.shape)
    if layer_indices is None:
        layer_indices = tuple(range(layers))
    if len(layer_indices) != layers:
        raise ValueError("layer_indices must contain one value for each score layer")

    result: list[LayerPlan] = []
    target = math.ceil(keep * experts * width)
    for position in range(layers):
        order = torch.argsort(scores[position].reshape(-1), descending=True, stable=True)
        raw_mask = torch.zeros(experts * width, dtype=torch.bool)
        raw_mask[order[:target]] = True
        raw_mask = raw_mask.view(experts, width)
        widths = _aligned_widths(raw_mask.sum(dim=1), block_size, width)

        channel_lists: list[tuple[int, ...]] = []
        survivors: list[int] = []
        for expert in range(experts):
            expert_width = int(widths[expert])
            if expert_width == 0:
                continue
            expert_order = torch.argsort(scores[position, expert], descending=True, stable=True)[
                :expert_width
            ]
            channels = tuple(sorted(int(index) for index in expert_order.tolist()))
            survivors.append(expert)
            channel_lists.append(channels)
        if len(survivors) < top_k:
            raise ValueError(
                f"layer {layer_indices[position]} keeps {len(survivors)} experts, "
                f"but the router needs at least {top_k}"
            )
        result.append(
            LayerPlan(
                layer=layer_indices[position],
                experts=tuple(survivors),
                channels=tuple(channel_lists),
            )
        )
    return PruningPlan(
        strategy="winnow",
        keep=keep,
        original_experts=experts,
        original_width=width,
        top_k=top_k,
        layers=tuple(result),
        block_size=block_size,
    )


def select_experts(
    scores: torch.Tensor,
    keep: float,
    *,
    top_k: int,
    original_width: int,
    layer_indices: tuple[int, ...] | None = None,
) -> PruningPlan:
    """Select whole experts by a stable per-layer REAP rank."""
    keep = validate_keep(keep)
    if scores.ndim != 2:
        raise ValueError("expert scores must have shape [layers, experts]")
    if not torch.isfinite(scores).all():
        raise ValueError("expert scores must be finite")
    layers, experts = map(int, scores.shape)
    keep_count = math.ceil(keep * experts)
    if keep_count < top_k:
        raise ValueError(
            f"keep retains {keep_count} experts, but the router needs at least {top_k}"
        )
    if layer_indices is None:
        layer_indices = tuple(range(layers))
    if len(layer_indices) != layers:
        raise ValueError("layer_indices must contain one value for each score layer")

    result: list[LayerPlan] = []
    all_channels = tuple(range(original_width))
    for position in range(layers):
        order = torch.argsort(scores[position], descending=True, stable=True)
        survivors = tuple(sorted(int(index) for index in order[:keep_count].tolist()))
        result.append(
            LayerPlan(
                layer=layer_indices[position],
                experts=survivors,
                channels=tuple(all_channels for _ in survivors),
            )
        )
    return PruningPlan(
        strategy="reap",
        keep=keep,
        original_experts=experts,
        original_width=original_width,
        top_k=top_k,
        layers=tuple(result),
    )
