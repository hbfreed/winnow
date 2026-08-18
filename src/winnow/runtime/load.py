"""Load a pruned Laguna checkpoint into the fused runtime for serving.

The pruned keep-50 Laguna-S checkpoint is ~118GB in BF16 — larger than system
RAM — so the loader materializes it layer by layer: each layer's ragged expert
tensors are packed into a :class:`~winnow.runtime.fast.FastLagunaMoE` block and
(optionally) quantized to INT8 W8A16 on the spot, which shrinks the experts 2×
before the next layer is read.  The finished model is then pipeline-split
across the visible GPUs with accelerate's ``dispatch_model``.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn


def _layer_bytes(module: nn.Module) -> int:
    total = 0
    for tensor in list(module.parameters()) + list(module.buffers()):
        total += tensor.numel() * tensor.dtype.itemsize
    return total


def _balanced_device_map(model: nn.Module, devices: list[str]) -> dict[str, str]:
    """Assign whole decoder layers to devices, balancing by byte size."""
    sizes = [(f"model.layers.{i}", _layer_bytes(layer)) for i, layer in enumerate(model.model.layers)]
    total = sum(size for _name, size in sizes)
    head_tail = _layer_bytes(model.model.embed_tokens) + _layer_bytes(model.lm_head)
    per_device = (total + head_tail) / len(devices)

    device_map: dict[str, str] = {
        "model.embed_tokens": devices[0],
        "model.rotary_emb": devices[0],
        "model.norm": devices[-1],
        "lm_head": devices[-1],
    }
    filled = float(_layer_bytes(model.model.embed_tokens))
    index = 0
    for name, size in sizes:
        if filled + size > per_device * 1.02 and index < len(devices) - 1:
            index += 1
            filled = 0.0
        device_map[name] = devices[index]
        filled += size
    return device_map


def load_fast_laguna(
    checkpoint: str | Path,
    *,
    int8: bool = True,
    devices: list[str] | None = None,
    dtype: torch.dtype = torch.bfloat16,
) -> nn.Module:
    """Build a pruned Laguna model on the fused runtime, ready to generate."""
    from accelerate import dispatch_model, init_empty_weights

    from ..stream import ShardReader, load_config, _sparse_positions
    from .fast import FastLagunaMoE
    from .laguna import WinnowLagunaForCausalLM

    checkpoint = Path(checkpoint)
    config = load_config(checkpoint)
    if config.model_type not in ("winnow_laguna", "laguna"):
        raise ValueError(f"expected a (pruned) Laguna checkpoint, got {config.model_type!r}")
    if getattr(config, "expert_widths", None) is None:
        raise ValueError("the fused loader expects a Winnow-pruned checkpoint")
    with init_empty_weights(include_buffers=False):
        model = WinnowLagunaForCausalLM(config)

    reader = ShardReader(checkpoint)
    positions = _sparse_positions(config)

    consumed_prefixes: list[str] = []
    for layer_idx, layer in enumerate(model.model.layers):
        if not hasattr(layer.mlp, "experts"):
            continue
        prefix = f"model.layers.{layer_idx}.mlp.experts."
        consumed_prefixes.append(prefix)
        widths = list(config.expert_widths[positions[layer_idx]])
        fast = FastLagunaMoE(
            config.hidden_size,
            widths,
            config.num_experts_per_tok,
            float(getattr(config, "moe_routed_scaling_factor", 1.0)),
        )
        fast = fast.to(dtype)
        with torch.no_grad():
            fast.gate.weight = nn.Parameter(
                reader.get(f"model.layers.{layer_idx}.mlp.gate.weight").to(dtype),
                requires_grad=False,
            )
            fast.e_score_correction_bias.data = reader.get(
                f"model.layers.{layer_idx}.mlp.gate.e_score_correction_bias"
            ).float()
            for slot, width in enumerate(widths):
                gate_up = reader.get(f"{prefix}gate_up_projs.{slot}").to(dtype)
                fast.load_expert_weight_(slot, "gate", gate_up[:width])
                fast.load_expert_weight_(slot, "up", gate_up[width:])
                fast.load_expert_weight_(
                    slot, "down", reader.get(f"{prefix}down_projs.{slot}").to(dtype)
                )
        if int8:
            fast.quantize_int8_()
        fast.shared_experts = layer.mlp.shared_experts
        # The fused block replaces both the router and the ragged experts; the
        # (still meta) shared expert loads through the state-dict path below.
        layer.mlp = fast

    # Everything that is not a routed expert or router loads by name (the
    # router weights were consumed by the fused blocks above).
    state: dict[str, torch.Tensor] = {}
    for name in reader.names:
        if any(name.startswith(prefix) for prefix in consumed_prefixes):
            continue
        if name.endswith((".mlp.gate.weight", ".mlp.gate.e_score_correction_bias")):
            continue
        target = name.replace(".mlp.shared_expert.", ".mlp.shared_experts.")
        state[target] = reader.get(name).to(dtype)
    reader.close()
    _missing, unexpected = model.load_state_dict(state, strict=False, assign=True)
    remaining_meta = [
        name
        for name, parameter in model.named_parameters()
        if parameter.is_meta and ".mlp." not in name
    ]
    if unexpected or remaining_meta:
        raise RuntimeError(
            f"fused load mismatch: unexpected={unexpected[:5]}, meta={remaining_meta[:5]}"
        )

    model.eval()
    if devices is None:
        devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())] or ["cpu"]
    if devices == ["cpu"]:
        return model
    device_map = _balanced_device_map(model, devices)
    return dispatch_model(model, device_map=device_map)


__all__ = ["load_fast_laguna"]
