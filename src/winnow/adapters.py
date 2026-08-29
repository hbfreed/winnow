"""Model-family detection and fused MoE discovery."""

from __future__ import annotations

import re
from dataclasses import dataclass

from torch import nn


@dataclass(frozen=True)
class ModelAdapter:
    """The structural values that Winnow needs from one model family."""

    family: str
    layers: tuple[tuple[int, nn.Module], ...]
    original_experts: int
    original_width: int
    top_k: int
    # Sigmoid routers (Laguna, DeepSeek-style gates) return normalized top-k
    # weights directly; softmax routers need the weights recovered from logits.
    sigmoid_router: bool = False


def block_router(block: nn.Module) -> nn.Module:
    """Return one MoE block's routing module (``gate``, or Afmoe's ``router``)."""
    router = getattr(block, "gate", None)
    if router is None:
        router = getattr(block, "router", None)
    if router is None:
        raise ValueError("the MoE block has neither a gate nor a router module")
    return router


def _layer_index(name: str) -> int:
    matches = list(re.finditer(r"layers\.(\d+)", name))
    if not matches:
        raise ValueError(f"cannot find a decoder layer index in {name!r}")
    return int(matches[-1].group(1))


def fused_moe_layers(model: nn.Module) -> tuple[tuple[int, nn.Module], ...]:
    """Find decoder MoE blocks that use fused three-dimensional expert weights."""
    found: list[tuple[int, str, nn.Module]] = []
    for name, module in model.named_modules():
        experts = getattr(module, "experts", None)
        if (
            (
                getattr(module, "gate", None) is not None
                or getattr(module, "router", None) is not None
            )
            and experts is not None
            and hasattr(experts, "gate_up_proj")
            and hasattr(experts, "down_proj")
        ):
            found.append((_layer_index(name), name, module))
    if not found:
        raise ValueError("Winnow did not find a supported fused MoE decoder block")
    found.sort(key=lambda item: item[0])
    indices = [item[0] for item in found]
    if len(indices) != len(set(indices)):
        raise ValueError("Winnow found more than one fused MoE block in a decoder layer")
    return tuple((index, module) for index, _name, module in found)


def adapter_for(model: nn.Module) -> ModelAdapter:
    """Return the adapter values for a supported Transformers model."""
    config = getattr(model, "config", None)
    model_type = str(getattr(config, "model_type", ""))
    if model_type == "olmoe":
        family = "olmoe"
    elif model_type in {"qwen3_5_moe", "qwen3_5_moe_text"}:
        family = "qwen3_5_moe"
    elif model_type == "laguna":
        family = "laguna"
    elif model_type == "afmoe":
        family = "afmoe"
    else:
        raise ValueError(
            f"unsupported model type {model_type!r}; "
            "supported types are olmoe, qwen3_5_moe, laguna, and afmoe"
        )

    layers = fused_moe_layers(model)
    first = layers[0][1]
    experts = int(first.experts.gate_up_proj.shape[0])
    width = int(first.experts.down_proj.shape[2])
    top_k = int(block_router(first).top_k)
    for layer_index, block in layers:
        shape = block.experts.gate_up_proj.shape
        down_shape = block.experts.down_proj.shape
        if int(shape[0]) != experts or int(shape[1]) != 2 * width:
            raise ValueError(f"layer {layer_index} has an inconsistent expert shape")
        if int(down_shape[0]) != experts or int(down_shape[2]) != width:
            raise ValueError(f"layer {layer_index} has an inconsistent down projection")
        if int(block_router(block).top_k) != top_k:
            raise ValueError(f"layer {layer_index} has a different router top-k")
    return ModelAdapter(
        family, layers, experts, width, top_k, sigmoid_router=family in ("laguna", "afmoe")
    )
