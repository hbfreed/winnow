"""Write and load structural Winnow checkpoints."""

from __future__ import annotations

import json
import shutil
from importlib.resources import files
from pathlib import Path
from typing import Any

import torch
from accelerate import init_empty_weights
from torch import nn

from .adapters import adapter_for
from .plan import PruningPlan
from .runtime import (
    WinnowOlmoeConfig,
    WinnowOlmoeForCausalLM,
    WinnowQwen3_5MoeConfig,
    WinnowQwen3_5MoeForCausalLM,
)


def _backbone(model: nn.Module) -> nn.Module:
    base = model.model
    return getattr(base, "language_model", None) or base


def _source_config(model: nn.Module, family: str):
    if family == "qwen3_5_moe":
        return getattr(model.config, "text_config", None) or model.config
    return model.config


def _runtime_types(family: str):
    if family == "olmoe":
        return WinnowOlmoeConfig, WinnowOlmoeForCausalLM
    if family == "qwen3_5_moe":
        return WinnowQwen3_5MoeConfig, WinnowQwen3_5MoeForCausalLM
    raise ValueError(f"unsupported model family {family!r}")


def _runtime_names(family: str) -> tuple[str, str, str, str]:
    if family == "olmoe":
        return (
            "WinnowOlmoeConfig",
            "WinnowOlmoeForCausalLM",
            "configuration_winnow_olmoe.py",
            "modeling_winnow_olmoe.py",
        )
    return (
        "WinnowQwen3_5MoeConfig",
        "WinnowQwen3_5MoeForCausalLM",
        "configuration_winnow_qwen3_5_moe.py",
        "modeling_winnow_qwen3_5_moe.py",
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
            survivor_tensor = torch.tensor(
                layer_plan.experts, dtype=torch.long, device=block.gate.weight.device
            )
            state[f"model.layers.{layer_plan.layer}.mlp.gate.weight"] = (
                block.gate.weight.index_select(0, survivor_tensor).contiguous()
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
    config_type, model_type = _runtime_types(adapter.family)
    config_name, architecture, config_file, model_file = _runtime_names(adapter.family)
    base = _source_config(model, adapter.family).to_dict()
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
    return output
