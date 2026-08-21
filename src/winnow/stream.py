"""Layer-streaming calibration and evaluation for models larger than memory.

The streamed path currently supports the Laguna family only: layers are
rebuilt with ``LagunaDecoderLayer``, so other families must go through the
in-memory ``prune`` path.

The streamed pass never holds more than one decoder layer of weights per
device.  The residual stream for every calibration sequence is kept in two
disk-backed ping-pong buffers per rank: each sweep step reads the layer input
from one buffer, runs the layer on the rank's device, and writes the layer
output to the other buffer.  Score statistics are additive across ranks, so
each rank calibrates on its own shard of the sequences and the driver sums the
per-rank statistics afterwards.

Held-out evaluation sequences ride along in the same buffers, after the
calibration sequences; they are excluded from score collection and produce a
perplexity from the final ``norm``/``lm_head`` step, which is how a pruned
checkpoint that cannot be loaded whole is evaluated at all.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from .adapters import ModelAdapter
from .collect import StatsCollector

_ADDITIVE_KEYS = ("token_count", "reap_sum", "channel_sum")


def load_config(checkpoint: str | Path):
    """Load a checkpoint config without going through remote-code resolution.

    Winnow's own pruned checkpoints carry an ``auto_map`` for end users, but
    here the runtime class is simply imported, which avoids the interactive
    trust-remote-code prompt in scripted runs.
    """
    from transformers import AutoConfig

    config_file = Path(checkpoint) / "config.json"
    model_type = None
    if config_file.exists():
        model_type = json.loads(config_file.read_text()).get("model_type")
    if model_type and model_type.startswith("winnow_"):
        from . import runtime

        by_model_type = {
            cls.model_type: cls
            for cls in (
                runtime.WinnowLagunaConfig,
                runtime.WinnowOlmoeConfig,
                runtime.WinnowQwen3_5MoeConfig,
            )
        }
        config = by_model_type[model_type].from_pretrained(checkpoint)
    else:
        config = AutoConfig.from_pretrained(checkpoint)
    config._attn_implementation = "sdpa"
    return config


# ---------------------------------------------------------------------------
# Checkpoint reading


def _tensor_index(checkpoint: Path) -> dict[str, Path]:
    """Map every tensor name to the shard file that stores it."""
    index_file = checkpoint / "model.safetensors.index.json"
    if index_file.exists():
        weight_map = json.loads(index_file.read_text())["weight_map"]
        return {name: checkpoint / file for name, file in weight_map.items()}
    single = checkpoint / "model.safetensors"
    if not single.exists():
        raise FileNotFoundError(f"no safetensors weights under {checkpoint}")
    from safetensors import safe_open

    with safe_open(single, framework="pt") as handle:
        return {name: single for name in handle.keys()}  # noqa: SIM118 — safe_open is not a dict


class ShardReader:
    """Read tensors by name, caching one open handle per shard file."""

    def __init__(self, checkpoint: str | Path) -> None:
        self.checkpoint = Path(checkpoint)
        self.index = _tensor_index(self.checkpoint)
        self._handles: dict[Path, object] = {}

    @property
    def names(self) -> list[str]:
        return list(self.index)

    def _handle(self, name: str):
        from safetensors import safe_open

        file = self.index[name]
        handle = self._handles.get(file)
        if handle is None:
            handle = safe_open(file, framework="pt")
            self._handles[file] = handle
        return handle

    def get(self, name: str, device: str = "cpu") -> torch.Tensor:
        return self._handle(name).get_tensor(name).to(device)

    def get_slice(self, name: str):
        """Return a lazy slice for one tensor; indexing it reads only that part."""
        return self._handle(name).get_slice(name)

    def close(self) -> None:
        self._handles.clear()


# ---------------------------------------------------------------------------
# Layer materialization


def _sparse_positions(config) -> dict[int, int]:
    """Map decoder layer index -> ordinal position among sparse layers."""
    positions = {}
    for layer_idx, kind in enumerate(config.mlp_layer_types):
        if kind == "sparse":
            positions[layer_idx] = len(positions)
    return positions


def _layer_name_groups(names: list[str]) -> dict[int, list[str]]:
    """Group tensor names by decoder layer index, once per sweep."""
    groups: dict[int, list[str]] = {}
    for name in names:
        match = re.match(r"model\.layers\.(\d+)\.", name)
        if match:
            groups.setdefault(int(match.group(1)), []).append(name)
    return groups


def _map_layer_state(
    config,
    layer_idx: int,
    reader: ShardReader,
    device: str,
    dtype: torch.dtype,
    names: list[str] | None = None,
) -> dict[str, torch.Tensor]:
    """Build one decoder layer's state dict, fusing per-expert shards if needed."""
    prefix = f"model.layers.{layer_idx}."
    if names is None:
        names = [name for name in reader.names if name.startswith(prefix)]
    state: dict[str, torch.Tensor] = {}
    expert_parts: dict[int, dict[str, torch.Tensor]] = {}
    for name in names:
        local = name.removeprefix(prefix)
        # Poolside checkpoints predate the native module: shared expert is
        # singular and the routing bias lives on the experts module.
        target = local.replace(".shared_expert.", ".shared_experts.").replace(
            "mlp.shared_expert.", "mlp.shared_experts."
        )
        if target == "mlp.experts.e_score_correction_bias":
            target = "mlp.gate.e_score_correction_bias"
        parts = target.split(".")
        if len(parts) >= 4 and parts[0] == "mlp" and parts[1] == "experts" and parts[2].isdigit():
            expert = int(parts[2])
            expert_parts.setdefault(expert, {})[parts[3]] = reader.get(name).to(dtype)
            continue
        state[target] = reader.get(name, device).to(dtype)

    if expert_parts:
        # Pack the per-expert shards on CPU and move each fused tensor in one
        # transfer; hundreds of small host-to-device copies serialize layer load.
        experts = len(expert_parts)
        width = expert_parts[0]["gate_proj"].shape[0]
        hidden = expert_parts[0]["gate_proj"].shape[1]
        gate_up = torch.empty(experts, 2 * width, hidden, dtype=dtype)
        down = torch.empty(experts, hidden, width, dtype=dtype)
        for expert, parts_dict in expert_parts.items():
            gate_up[expert, :width] = parts_dict["gate_proj"]
            gate_up[expert, width:] = parts_dict["up_proj"]
            down[expert] = parts_dict["down_proj"]
        state["mlp.experts.gate_up_proj"] = gate_up.to(device)
        state["mlp.experts.down_proj"] = down.to(device)
    return state


def materialize_layer(
    config,
    layer_idx: int,
    reader: ShardReader,
    device: str,
    dtype: torch.dtype,
    names: list[str] | None = None,
) -> nn.Module:
    """Load one decoder layer's weights onto ``device`` and return the module."""
    from transformers.models.laguna.modeling_laguna import LagunaDecoderLayer

    with torch.device("meta"):
        layer = LagunaDecoderLayer(config, layer_idx)

    widths_table = getattr(config, "expert_widths", None)
    if widths_table is not None and hasattr(layer.mlp, "experts"):
        from .runtime.ragged import install_ragged_experts

        widths = widths_table[_sparse_positions(config)[layer_idx]]
        install_ragged_experts(layer.mlp, list(widths), config.hidden_size, config.hidden_act)

    state = _map_layer_state(config, layer_idx, reader, device, dtype, names)
    layer.load_state_dict(state, strict=True, assign=True)
    return layer.eval()


# ---------------------------------------------------------------------------
# Disk-backed residual buffers


def _open_buffer(path: Path, shape: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
    """Memory-map ``path`` as a tensor, creating or resizing the file first."""
    numel = math.prod(shape)
    nbytes = numel * dtype.itemsize
    with open(path, "a+b") as handle:
        handle.truncate(nbytes)
    storage = torch.from_file(str(path), shared=True, size=numel, dtype=dtype)
    return storage.view(*shape)


@dataclass
class RankData:
    """Paths and sizes for one rank's shard of the streamed run."""

    directory: Path
    calibration_sequences: int
    eval_sequences: int
    sequence_length: int
    hidden_size: int
    buffer_dtype: torch.dtype = torch.bfloat16

    @property
    def total(self) -> int:
        return self.calibration_sequences + self.eval_sequences

    def buffer(self, which: str) -> torch.Tensor:
        return _open_buffer(
            self.directory / f"residual_{which}.bin",
            (self.total, self.sequence_length, self.hidden_size),
            self.buffer_dtype,
        )

    def token_ids(self) -> torch.Tensor:
        return _open_buffer(
            self.directory / "token_ids.bin",
            (self.total, self.sequence_length),
            torch.int32,
        )


# ---------------------------------------------------------------------------
# The sweep


def _layer_inputs(config, device: str, dtype: torch.dtype, batch: torch.Tensor):
    """Build position embeddings and per-layer-type masks for one chunk."""
    from transformers.masking_utils import (
        create_causal_mask,
        create_sliding_window_causal_mask,
    )
    from transformers.models.laguna.modeling_laguna import LagunaRotaryEmbedding

    position_ids = torch.arange(batch.shape[1], device=device).unsqueeze(0)
    rotary = LagunaRotaryEmbedding(config).to(device)
    layer_types = set(config.layer_types)
    position_embeddings = {
        layer_type: rotary(batch, position_ids, layer_type) for layer_type in layer_types
    }
    mask_kwargs = {
        "config": config,
        "inputs_embeds": batch,
        "attention_mask": None,
        "past_key_values": None,
        "position_ids": position_ids,
    }
    creators = {
        "full_attention": lambda: create_causal_mask(**mask_kwargs),
        "sliding_attention": lambda: create_sliding_window_causal_mask(**mask_kwargs),
    }
    masks = {layer_type: creators[layer_type]() for layer_type in layer_types}
    return position_embeddings, masks


def _chunks(total: int, size: int):
    for start in range(0, total, size):
        yield start, min(start + size, total)


def run_rank_sweep(
    rank_data: RankData,
    checkpoint: str | Path,
    *,
    device: str,
    dtype: torch.dtype = torch.bfloat16,
    collect_stats: bool = True,
    chunk_sequences: int = 4,
) -> dict:
    """Sweep every decoder layer over one rank's residual shard.

    Returns ``{"stats": ..., "loss_sum": float, "loss_tokens": int}`` where
    ``stats`` is ``None`` when ``collect_stats`` is off.
    """
    config = load_config(checkpoint)
    reader = ShardReader(checkpoint)
    positions = _sparse_positions(config)

    collector = None
    if collect_stats:
        if getattr(config, "expert_widths", None) is not None:
            raise ValueError("score collection expects the unpruned source checkpoint")
        adapter = ModelAdapter(
            family="laguna",
            layers=tuple((index, None) for index in sorted(positions)),
            original_experts=config.num_experts,
            original_width=config.moe_intermediate_size,
            top_k=config.num_experts_per_tok,
            sigmoid_router=True,
        )
        collector = StatsCollector(None, adapter)

    source = rank_data.buffer("a")
    target = rank_data.buffer("b")
    calib = rank_data.calibration_sequences
    total = rank_data.total
    # Never let one chunk span the calibration/held-out boundary, so the score
    # hooks can be dropped exactly when the held-out sequences begin.
    spans = list(_chunks(calib, chunk_sequences)) + [
        (calib + start, calib + stop) for start, stop in _chunks(total - calib, chunk_sequences)
    ]

    inputs_cache: dict[int, tuple] = {}
    layer_names = _layer_name_groups(reader.names)

    with torch.no_grad():
        for layer_idx in range(config.num_hidden_layers):
            layer = materialize_layer(
                config, layer_idx, reader, device, dtype, layer_names.get(layer_idx)
            )
            layer_type = config.layer_types[layer_idx]
            handles = []
            if collector is not None and layer_idx in positions:
                handles = collector.attach_block(positions[layer_idx], layer.mlp)
            for start, stop in spans:
                if handles and start >= calib:
                    for handle in handles:
                        handle.remove()
                    handles = []
                batch = source[start:stop].to(device=device, dtype=dtype)
                cached = inputs_cache.get(stop - start)
                if cached is None:
                    cached = _layer_inputs(config, device, dtype, batch)
                    inputs_cache[stop - start] = cached
                position_embeddings, masks = cached
                out = layer(
                    batch,
                    attention_mask=masks[layer_type],
                    position_embeddings=position_embeddings[layer_type],
                )
                target[start:stop].copy_(out)
            for handle in handles:
                handle.remove()
            # No empty_cache here: consecutive layers allocate the same shapes,
            # so the caching allocator reuses the freed blocks directly.
            del layer
            source, target = target, source

        # Final norm + lm_head: cross-entropy on the held-out sequences.
        loss_sum = 0.0
        loss_tokens = 0
        if rank_data.eval_sequences:
            from transformers.models.laguna.modeling_laguna import LagunaRMSNorm

            with torch.device("meta"):
                norm = LagunaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            norm.load_state_dict(
                {"weight": reader.get("model.norm.weight", device).to(dtype)},
                strict=True,
                assign=True,
            )
            lm_head = reader.get("lm_head.weight", device).to(dtype)
            token_ids = rank_data.token_ids()
            for start, stop in spans:
                if stop <= calib:
                    continue
                lo = max(start, calib)
                hidden = source[lo:stop].to(device=device, dtype=dtype)
                labels = token_ids[lo:stop].to(device).long()
                # One sequence at a time keeps the full-vocab logits (and their
                # float32 copy) to one [sequence, vocab] tensor; the last
                # position has no label, so it skips the matmul entirely.
                for row_hidden, row_labels in zip(hidden, labels):
                    logits = F.linear(norm(row_hidden[:-1]), lm_head).float()
                    loss = F.cross_entropy(logits, row_labels[1:], reduction="sum")
                    loss_sum += float(loss)
                    loss_tokens += row_labels[1:].numel()

    reader.close()
    stats = collector.stats if collector is not None else None
    return {"stats": stats, "loss_sum": loss_sum, "loss_tokens": loss_tokens}


# ---------------------------------------------------------------------------
# Driver


def _write_rank_shards(
    checkpoint: Path,
    sequences: torch.Tensor,
    calibration_sequences: int,
    workdir: Path,
    ranks: int,
    dtype: torch.dtype,
) -> list[RankData]:
    """Embed all sequences on CPU and split them into per-rank buffers."""
    config = load_config(checkpoint)
    reader = ShardReader(checkpoint)
    embeddings = reader.get("model.embed_tokens.weight")
    reader.close()

    eval_sequences = sequences.shape[0] - calibration_sequences

    def _split(total: int) -> list[int]:
        base, extra = divmod(total, ranks)
        return [base + (rank < extra) for rank in range(ranks)]

    calib_split = _split(calibration_sequences)
    eval_split = _split(eval_sequences)

    rank_data = []
    calib_start = 0
    eval_start = calibration_sequences
    for rank in range(ranks):
        directory = workdir / f"rank{rank}"
        directory.mkdir(parents=True, exist_ok=True)
        data = RankData(
            directory=directory,
            calibration_sequences=calib_split[rank],
            eval_sequences=eval_split[rank],
            sequence_length=int(sequences.shape[1]),
            hidden_size=config.hidden_size,
            buffer_dtype=dtype,
        )
        rows = torch.cat(
            [
                sequences[calib_start : calib_start + calib_split[rank]],
                sequences[eval_start : eval_start + eval_split[rank]],
            ]
        )
        buffer = data.buffer("a")
        token_ids = data.token_ids()
        token_ids[:] = rows.to(torch.int32)
        for index in range(rows.shape[0]):
            buffer[index] = F.embedding(rows[index], embeddings)
        # Create the ping-pong partner up front so workers never resize files.
        data.buffer("b")
        rank_data.append(data)
        calib_start += calib_split[rank]
        eval_start += eval_split[rank]
    return rank_data


def _spawn_worker(rank: int, rank_data: list[RankData], checkpoint: str, kwargs: dict) -> None:
    result = run_rank_sweep(rank_data[rank], checkpoint, device=f"cuda:{rank}", **kwargs)
    torch.save(result, rank_data[rank].directory / "result.pt")


def merge_stats(parts: list[dict]) -> dict:
    """Sum per-rank statistics into one stats dict."""
    merged = {
        key: value.clone() if torch.is_tensor(value) else value for key, value in parts[0].items()
    }
    for part in parts[1:]:
        merged["n_tokens"] += part["n_tokens"]
        for key in _ADDITIVE_KEYS:
            merged[key] += part[key]
    return merged


def stream_run(
    checkpoint: str | Path,
    sequences: torch.Tensor,
    *,
    calibration_sequences: int,
    workdir: str | Path,
    ranks: int = 1,
    device: str | None = None,
    dtype: torch.dtype = torch.bfloat16,
    collect_stats: bool = True,
    chunk_sequences: int = 4,
) -> dict:
    """Run a full layer-streamed pass and return merged stats and perplexity.

    ``sequences`` is ``[N, S]`` token ids; the first ``calibration_sequences``
    rows are scored, the rest are held out for perplexity.
    """
    checkpoint = Path(checkpoint)
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    rank_data = _write_rank_shards(
        checkpoint, sequences, calibration_sequences, workdir, ranks, dtype
    )
    kwargs = {
        "dtype": dtype,
        "collect_stats": collect_stats,
        "chunk_sequences": chunk_sequences,
    }

    if ranks == 1:
        only = run_rank_sweep(
            rank_data[0],
            checkpoint,
            device=device or ("cuda:0" if torch.cuda.is_available() else "cpu"),
            **kwargs,
        )
        results = [only]
    else:
        import torch.multiprocessing as mp

        mp.spawn(
            _spawn_worker,
            args=(rank_data, str(checkpoint), kwargs),
            nprocs=ranks,
            join=True,
        )
        results = [
            torch.load(data.directory / "result.pt", weights_only=False) for data in rank_data
        ]

    # The residual and token buffers are pure scratch and can be tens of
    # gigabytes; drop them as soon as the per-rank results are in.
    for data in rank_data:
        for scratch in ("residual_a.bin", "residual_b.bin", "token_ids.bin"):
            (data.directory / scratch).unlink(missing_ok=True)

    loss_sum = sum(result["loss_sum"] for result in results)
    loss_tokens = sum(result["loss_tokens"] for result in results)
    perplexity = math.exp(loss_sum / loss_tokens) if loss_tokens else None
    stats = None
    if collect_stats:
        stats = merge_stats([result["stats"] for result in results])
    return {"stats": stats, "perplexity": perplexity, "loss_tokens": loss_tokens}
