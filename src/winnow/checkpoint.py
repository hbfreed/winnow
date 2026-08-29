"""Write and load structural Winnow checkpoints."""

from __future__ import annotations

import json
import re
import shutil
from importlib.resources import files
from pathlib import Path
from typing import Any

import torch
from accelerate import init_empty_weights
from torch import nn

from .adapters import adapter_for, block_router
from .plan import PruningPlan
from .runtime import (
    WinnowAfmoeConfig,
    WinnowAfmoeForCausalLM,
    WinnowLagunaConfig,
    WinnowLagunaForCausalLM,
    WinnowOlmoeConfig,
    WinnowOlmoeForCausalLM,
    WinnowQwen3_5MoeConfig,
    WinnowQwen3_5MoeForCausalLM,
)


def _backbone(model: nn.Module) -> nn.Module:
    base = model.model
    return getattr(base, "language_model", None) or base


def _source_config(model: nn.Module):
    return getattr(model.config, "text_config", None) or model.config


FAMILIES = {
    "olmoe": (WinnowOlmoeConfig, WinnowOlmoeForCausalLM),
    "qwen3_5_moe": (WinnowQwen3_5MoeConfig, WinnowQwen3_5MoeForCausalLM),
    "laguna": (WinnowLagunaConfig, WinnowLagunaForCausalLM),
    "afmoe": (WinnowAfmoeConfig, WinnowAfmoeForCausalLM),
}

# Per-family router tensor names, relative to one decoder layer's ``mlp``:
# the routing weight and the aux-loss-free selection bias candidates in
# checkpoint order of preference (the first name is the one written).
_ROUTER_NAMES = {
    "laguna": ("gate.weight", ("gate.e_score_correction_bias", "experts.e_score_correction_bias")),
    "afmoe": ("router.gate.weight", ("expert_bias",)),
}


def _runtime_types(family: str):
    try:
        return FAMILIES[family]
    except KeyError:
        raise ValueError(f"unsupported model family {family!r}") from None


def _runtime_names(family: str) -> tuple[str, str, str, str]:
    config_type, model_type = _runtime_types(family)
    stem = config_type.model_type  # e.g. "winnow_olmoe" — the shim filenames follow it
    return (
        config_type.__name__,
        model_type.__name__,
        f"configuration_{stem}.py",
        f"modeling_{stem}.py",
    )


def extract_state(model: nn.Module, plan: PruningPlan) -> dict[str, torch.Tensor]:
    """Create the exact ragged runtime state from a fused source model."""
    adapter = adapter_for(model)
    if len(plan.layers) != len(adapter.layers):
        raise ValueError("the plan does not match the number of model layers")
    backbone = _backbone(model)
    state: dict[str, torch.Tensor] = {}
    for key, value in backbone.state_dict().items():
        if ".mlp.experts." in key or key.endswith(".mlp.gate.weight"):
            continue
        state[f"model.{key}"] = value

    blocks = dict(adapter.layers)
    with torch.no_grad():
        for layer_plan in plan.layers:
            block = blocks[layer_plan.layer]
            source = block.experts
            router = block_router(block)
            weight = router.weight if hasattr(router, "weight") else router.gate.weight
            weight_name, bias_names = _ROUTER_NAMES.get(adapter.family, ("gate.weight", ()))
            survivor_tensor = torch.tensor(
                layer_plan.experts, dtype=torch.long, device=weight.device
            )
            state[f"model.layers.{layer_plan.layer}.mlp.{weight_name}"] = weight.index_select(
                0, survivor_tensor
            ).contiguous()
            if bias_names:
                bias = getattr(router, "e_score_correction_bias", None)
                if bias is None:
                    bias = getattr(block, "expert_bias", None)
                if bias is not None:
                    state[f"model.layers.{layer_plan.layer}.mlp.{bias_names[0]}"] = (
                        bias.index_select(0, survivor_tensor.to(bias.device)).contiguous()
                    )
            for slot, (expert, channels) in enumerate(
                zip(layer_plan.experts, layer_plan.channels, strict=True)
            ):
                channel_tensor = torch.tensor(
                    channels, dtype=torch.long, device=source.gate_up_proj.device
                )
                rows = torch.cat([channel_tensor, channel_tensor + adapter.original_width])
                state[f"model.layers.{layer_plan.layer}.mlp.experts.gate_up_projs.{slot}"] = (
                    source.gate_up_proj[expert].index_select(0, rows).contiguous()
                )
                state[f"model.layers.{layer_plan.layer}.mlp.experts.down_projs.{slot}"] = (
                    source.down_proj[expert].index_select(1, channel_tensor).contiguous()
                )
        state["lm_head.weight"] = model.lm_head.weight
    return state


def build_runtime_config(family: str, source_config_dict: dict, plan: PruningPlan):
    """Build the pruned-runtime configuration for one plan."""
    config_type, _model_type = _runtime_types(family)
    config_name, architecture, config_file, model_file = _runtime_names(family)
    base = dict(source_config_dict)
    for key in ("model_type", "architectures", "auto_map", "transformers_version"):
        base.pop(key, None)
    config = config_type(
        expert_widths=[list(layer.widths) for layer in plan.layers],
        expert_indices=[list(layer.experts) for layer in plan.layers],
        **base,
    )
    config.architectures = [architecture]
    config.auto_map = {
        "AutoConfig": f"{config_file[:-3]}.{config_name}",
        "AutoModelForCausalLM": f"{model_file[:-3]}.{architecture}",
    }
    config.winnow = {
        "schema_version": 1,
        "strategy": plan.strategy,
        "keep": plan.keep,
    }
    return config


def _write_support_files(
    output: Path, family: str, plan: PruningPlan, metadata: dict[str, Any], tokenizer
) -> None:
    """Write the shim modules, ``winnow.json``, and tokenizer files."""
    _config_name, _architecture, config_file, model_file = _runtime_names(family)
    if tokenizer is not None:
        tokenizer.save_pretrained(output)
    code = files("winnow.checkpoint_code")
    for filename in (config_file, model_file):
        shutil.copyfile(code.joinpath(filename), output / filename)
    complete_metadata = dict(metadata)
    complete_metadata["schema_version"] = 1
    complete_metadata["plan"] = plan.to_dict()
    (output / "winnow.json").write_text(
        json.dumps(complete_metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def save_checkpoint(
    model: nn.Module,
    plan: PruningPlan,
    output: str | Path,
    *,
    metadata: dict[str, Any],
    tokenizer=None,
    max_shard_size: str = "5GB",
) -> Path:
    """Write a self-contained Hugging Face checkpoint and ``winnow.json``."""
    adapter = adapter_for(model)
    _config_type, model_type = _runtime_types(adapter.family)
    config = build_runtime_config(adapter.family, _source_config(model).to_dict(), plan)

    state = extract_state(model, plan)
    with init_empty_weights(include_buffers=False):
        pruned = model_type(config)
    missing, unexpected = pruned.load_state_dict(state, strict=False, assign=True)
    if missing or unexpected:
        raise RuntimeError(f"checkpoint state mismatch: missing={missing}, unexpected={unexpected}")

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    pruned.save_pretrained(
        output,
        safe_serialization=True,
        max_shard_size=max_shard_size,
    )
    _write_support_files(output, adapter.family, plan, metadata, tokenizer)
    return output


class _ShardWriter:
    """Accumulate tensors and flush them as size-capped safetensors shards."""

    def __init__(self, output: Path, max_shard_bytes: int) -> None:
        self.output = output
        self.max_shard_bytes = max_shard_bytes
        self.pending: dict[str, torch.Tensor] = {}
        self.pending_bytes = 0
        self.total_bytes = 0
        self.files: list[Path] = []
        self.weight_map: dict[str, int] = {}

    def add(self, name: str, tensor: torch.Tensor) -> None:
        tensor = tensor.contiguous()
        self.pending[name] = tensor
        self.weight_map[name] = len(self.files)
        self.pending_bytes += tensor.numel() * tensor.dtype.itemsize
        self.total_bytes += tensor.numel() * tensor.dtype.itemsize
        if self.pending_bytes >= self.max_shard_bytes:
            self.flush()

    def flush(self) -> None:
        if not self.pending:
            return
        from safetensors.torch import save_file

        path = self.output / f"shard-{len(self.files):05d}.tmp"
        save_file(self.pending, path)
        self.files.append(path)
        self.pending = {}
        self.pending_bytes = 0

    def finalize(self) -> None:
        self.flush()
        count = len(self.files)
        if count == 1:
            names = {0: "model.safetensors"}
            self.files[0].rename(self.output / names[0])
        else:
            names = {i: f"model-{i + 1:05d}-of-{count:05d}.safetensors" for i in range(count)}
            for i, path in enumerate(self.files):
                path.rename(self.output / names[i])
            index = {
                "metadata": {"total_size": self.total_bytes},
                "weight_map": {name: names[shard] for name, shard in self.weight_map.items()},
            }
            (self.output / "model.safetensors.index.json").write_text(
                json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )


def save_checkpoint_streamed(
    source: str | Path,
    plan: PruningPlan,
    output: str | Path,
    *,
    metadata: dict[str, Any],
    tokenizer=None,
    max_shard_bytes: int = 4 * 2**30,
) -> Path:
    """Write a pruned checkpoint straight from source shards, tensor by tensor.

    Unlike :func:`save_checkpoint` this never materializes the model, so it
    works for checkpoints far larger than memory.  Laguna only for now.
    """
    from .stream import ShardReader, load_config

    source = Path(source)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    source_config = load_config(source)
    family = str(source_config.model_type)
    if family not in _ROUTER_NAMES:
        raise ValueError("the streamed writer supports the laguna and afmoe families only")
    gate_local, bias_locals = _ROUTER_NAMES[family]
    config = build_runtime_config(family, source_config.to_dict(), plan)

    reader = ShardReader(source)
    plan_by_layer = {layer_plan.layer: layer_plan for layer_plan in plan.layers}
    width = plan.original_width
    by_layer: dict[int, list[str]] = {}
    other: list[str] = []
    for name in reader.names:
        match = re.search(r"^model\.layers\.(\d+)\.", name)
        if match:
            by_layer.setdefault(int(match.group(1)), []).append(name)
        else:
            other.append(name)

    writer = _ShardWriter(output, max_shard_bytes)
    if "model.embed_tokens.weight" in other:
        writer.add("model.embed_tokens.weight", reader.get("model.embed_tokens.weight"))

    def _rename(name: str) -> str:
        return name.replace(".mlp.shared_expert.", ".mlp.shared_experts.")

    for layer_idx in sorted(by_layer):
        names = by_layer[layer_idx]
        layer_plan = plan_by_layer.get(layer_idx)
        prefix = f"model.layers.{layer_idx}."
        if layer_plan is None:
            for name in names:
                writer.add(_rename(name), reader.get(name))
            continue

        survivors = torch.tensor(layer_plan.experts, dtype=torch.long)
        gate_weight = reader.get(f"{prefix}mlp.{gate_local}")
        writer.add(f"{prefix}mlp.{gate_local}", gate_weight.index_select(0, survivors))
        bias_names = tuple(f"{prefix}mlp.{local}" for local in bias_locals)
        bias = next(
            (reader.get(name) for name in bias_names if name in reader.index),
            torch.zeros(plan.original_experts, dtype=gate_weight.dtype),
        )
        writer.add(bias_names[0], bias.index_select(0, survivors))

        fused = f"{prefix}mlp.experts.gate_up_proj" in reader.index
        if fused:
            # Lazy slices: only the surviving experts' slabs are read from disk.
            gate_up_slice = reader.get_slice(f"{prefix}mlp.experts.gate_up_proj")
            down_slice = reader.get_slice(f"{prefix}mlp.experts.down_proj")
        for slot, (expert, channels) in enumerate(
            zip(layer_plan.experts, layer_plan.channels, strict=True)
        ):
            channel_tensor = torch.tensor(channels, dtype=torch.long)
            if fused:
                rows = torch.cat([channel_tensor, channel_tensor + width])
                gate_up = gate_up_slice[expert].index_select(0, rows)
                down = down_slice[expert].index_select(1, channel_tensor)
            else:
                gate = reader.get(f"{prefix}mlp.experts.{expert}.gate_proj.weight")
                up = reader.get(f"{prefix}mlp.experts.{expert}.up_proj.weight")
                gate_up = torch.cat(
                    [
                        gate.index_select(0, channel_tensor),
                        up.index_select(0, channel_tensor),
                    ]
                )
                down = reader.get(f"{prefix}mlp.experts.{expert}.down_proj.weight").index_select(
                    1, channel_tensor
                )
            writer.add(f"{prefix}mlp.experts.gate_up_projs.{slot}", gate_up)
            writer.add(f"{prefix}mlp.experts.down_projs.{slot}", down)

        consumed = {f"mlp.{local}" for local in (gate_local, *bias_locals)}
        for name in names:
            local = name.removeprefix(prefix)
            if ".mlp.experts." in name or local in consumed:
                continue
            writer.add(_rename(name), reader.get(name))

    for name in other:
        if name != "model.embed_tokens.weight":
            writer.add(name, reader.get(name))

    writer.finalize()
    reader.close()
    config.save_pretrained(output)
    _write_support_files(output, family, plan, metadata, tokenizer)
    return output
