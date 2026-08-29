"""Collect the sufficient statistics for channel pruning and REAP."""

from __future__ import annotations

from typing import Any, Self

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.hooks import RemovableHandle

from .adapters import ModelAdapter, adapter_for, block_router
from .runtime.ragged import routed_token_segments


class StatsCollector:
    """Collect both score types in one calibration pass."""

    def __init__(self, model: nn.Module | None, adapter: ModelAdapter | None = None) -> None:
        if model is None and adapter is None:
            raise ValueError("an adapter is required when no model is given")
        self.model = model
        self.adapter = adapter or adapter_for(model)
        layers = len(self.adapter.layers)
        experts = self.adapter.original_experts
        width = self.adapter.original_width
        self.stats: dict[str, Any] = {
            "n_tokens": 0,
            "token_count": torch.zeros(layers, experts, dtype=torch.int64),
            "reap_sum": torch.zeros(layers, experts, dtype=torch.float32),
            "channel_sum": torch.zeros(layers, experts, width, dtype=torch.float32),
            "down_norm": torch.zeros(layers, experts, width, dtype=torch.float32),
            "layer_indices": tuple(index for index, _block in self.adapter.layers),
        }
        self._handles: list[RemovableHandle] = []
        self._mask: torch.Tensor | None = None
        self._router: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        self._weights: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}

    def _capture_mask(self, _module: nn.Module, args: tuple, kwargs: dict) -> None:
        mask = kwargs.get("attention_mask")
        if mask is None and len(args) > 1 and isinstance(args[1], torch.Tensor):
            mask = args[1]
        if mask is None:
            self._mask = None
            return
        if mask.ndim != 2:
            raise RuntimeError("calibration attention masks must have shape [batch, sequence]")
        self._mask = mask.detach().reshape(-1).bool()

    def _gate_hook(self, position: int):
        def hook(_module: nn.Module, _args: tuple, output: Any) -> None:
            if not isinstance(output, tuple) or len(output) != 3:
                raise RuntimeError("the router output contract changed")
            logits, _weights, indices = output
            self._router[position] = (logits.detach(), indices.detach())

        return hook

    def _stash_hook(self, position: int, experts: nn.Module):
        def hook(_module: nn.Module, _args: tuple) -> None:
            if position not in self._weights and not experts.gate_up_proj.is_meta:
                self._weights[position] = (
                    experts.gate_up_proj.data,
                    experts.down_proj.data,
                )

        return hook

    def _experts_hook(self, position: int):
        def hook(module: nn.Module, args: tuple, kwargs: dict, _output: Any) -> None:
            hidden_states = args[0] if args else kwargs["hidden_states"]
            indices = args[1] if len(args) > 1 else kwargs["top_k_index"]
            actual_weights = args[2] if len(args) > 2 else kwargs["top_k_weights"]
            self._process(position, module, hidden_states, indices, actual_weights)

        return hook

    def _process(
        self,
        position: int,
        experts_module: nn.Module,
        hidden_states: torch.Tensor,
        indices: torch.Tensor,
        actual_weights: torch.Tensor,
    ) -> None:
        experts = self.adapter.original_experts
        width = self.adapter.original_width
        token_count = int(hidden_states.shape[0])
        mask = self._mask
        if mask is not None and mask.numel() != token_count:
            raise RuntimeError("the attention mask does not match the routed token count")
        if position == 0:
            self.stats["n_tokens"] += token_count if mask is None else int(mask.sum())

        if position not in self._router:
            raise RuntimeError("the experts ran without a matching router call")
        router_logits, router_indices = self._router.pop(position)
        if not torch.equal(router_indices.to(indices.device), indices):
            raise RuntimeError("the router indices do not match the expert inputs")

        stashed = self._weights.get(position)
        if stashed is None:
            gate_up = experts_module.gate_up_proj
            down = experts_module.down_proj
            if gate_up.is_meta or down.is_meta:
                raise RuntimeError("the expert weights were offloaded before score collection")
        else:
            gate_up, down = stashed

        device = hidden_states.device
        mask_device = None if mask is None else mask.to(device)
        actual_weights = actual_weights.to(device=device, dtype=torch.float32)
        if self.adapter.sigmoid_router:
            # A softmax over sigmoid-router logits is meaningless; the gate's
            # returned weights are already the normalized top-k routing weights
            # (renormalize to stay exact under float32).
            normalized_weights = actual_weights / actual_weights.sum(dim=-1, keepdim=True)
        else:
            probabilities = F.softmax(router_logits.to(device), dim=-1, dtype=torch.float32)
            normalized_weights = probabilities.gather(1, indices.to(device))
            normalized_weights /= normalized_weights.sum(dim=-1, keepdim=True)
        indices = indices.to(device)

        reap = torch.zeros(experts, dtype=torch.float32, device=device)
        channels = torch.zeros(experts, width, dtype=torch.float32, device=device)
        counts, segments = routed_token_segments(indices, experts, keep_mask=mask_device)
        with torch.no_grad():
            for expert, tokens, slots in segments:
                gate, up = F.linear(hidden_states[tokens], gate_up[expert]).chunk(2, dim=-1)
                activation = experts_module.act_fn(gate) * up
                output = F.linear(activation, down[expert])
                output_norm = torch.linalg.vector_norm(output, ord=2, dim=-1, dtype=torch.float32)
                reap[expert] = (normalized_weights[tokens, slots] * output_norm).sum()
                channels[expert] = (
                    actual_weights[tokens, slots, None] * activation.abs().float()
                ).sum(dim=0)

            self.stats["token_count"][position] += counts.cpu()
            self.stats["reap_sum"][position] += reap.cpu()
            self.stats["channel_sum"][position] += channels.cpu()
            if not bool(self.stats["down_norm"][position].any()):
                self.stats["down_norm"][position] = down.detach().float().norm(dim=1).cpu()

        self._weights.pop(position, None)
        if position == len(self.adapter.layers) - 1:
            self._mask = None

    def attach_block(self, position: int, block: nn.Module) -> list[RemovableHandle]:
        """Attach the calibration hooks for one MoE block at ``position``.

        Returns the handles so a layer-streaming caller can remove them before
        the block is discarded; they are also tracked for ``detach``.
        """
        handles = [
            block_router(block).register_forward_hook(self._gate_hook(position)),
            block.experts.register_forward_hook(self._experts_hook(position), with_kwargs=True),
        ]
        act_fn = block.experts.act_fn
        if isinstance(act_fn, nn.Module):
            handles.append(
                act_fn.register_forward_pre_hook(self._stash_hook(position, block.experts))
            )
        self._handles.extend(handles)
        return handles

    def attach(self) -> Self:
        """Attach the calibration hooks."""
        if self._handles:
            raise RuntimeError("the collector is already attached")
        if self.model is None:
            raise RuntimeError("hooks need a model; use attach_block for streamed layers")
        self._handles.append(
            self.model.register_forward_pre_hook(self._capture_mask, with_kwargs=True)
        )
        for position, (_layer_index, block) in enumerate(self.adapter.layers):
            self.attach_block(position, block)
        return self

    def detach(self) -> None:
        """Remove all hooks and transient values."""
        for handle in self._handles:
            handle.remove()
        self._handles = []
        self._mask = None
        self._router.clear()
        self._weights.clear()

    def __enter__(self) -> Self:
        return self.attach()

    def __exit__(self, _exc_type, _exc_value, _traceback) -> bool:
        self.detach()
        return False
