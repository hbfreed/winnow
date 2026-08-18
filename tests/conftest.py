import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

import pytest
import torch
from transformers import OlmoeConfig, OlmoeForCausalLM
from transformers.models.laguna.configuration_laguna import LagunaConfig
from transformers.models.laguna.modeling_laguna import LagunaForCausalLM
from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import (
    Qwen3_5MoeTextConfig,
)
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    Qwen3_5MoeForCausalLM,
)


@pytest.fixture()
def tiny_olmoe():
    torch.manual_seed(0)
    config = OlmoeConfig(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=8,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=32,
        num_experts=4,
        num_experts_per_tok=2,
        norm_topk_prob=False,
        eos_token_id=2,
        pad_token_id=0,
    )
    return OlmoeForCausalLM(config).eval()


@pytest.fixture()
def input_ids():
    generator = torch.Generator().manual_seed(4)
    return torch.randint(0, 64, (2, 8), generator=generator)


@pytest.fixture()
def tiny_qwen():
    torch.manual_seed(0)
    config = Qwen3_5MoeTextConfig(
        vocab_size=64,
        hidden_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        max_position_embeddings=32,
        layer_types=["full_attention", "full_attention"],
        moe_intermediate_size=8,
        shared_expert_intermediate_size=8,
        num_experts=4,
        num_experts_per_tok=2,
        eos_token_id=2,
        pad_token_id=0,
    )
    return Qwen3_5MoeForCausalLM(config).eval()


@pytest.fixture()
def tiny_laguna():
    torch.manual_seed(0)
    config = LagunaConfig(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=8,
        num_hidden_layers=3,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        max_position_embeddings=32,
        layer_types=["full_attention", "sliding_attention", "full_attention"],
        mlp_layer_types=["dense", "sparse", "sparse"],
        sliding_window=4,
        moe_intermediate_size=8,
        shared_expert_intermediate_size=8,
        num_experts=4,
        num_experts_per_tok=2,
        moe_routed_scaling_factor=2.5,
        eos_token_id=2,
        pad_token_id=0,
    )
    model = LagunaForCausalLM(config).eval()
    # Give the aux-loss-free routing bias real values so a checkpoint test that
    # mishandles its slicing changes the routing and fails the parity check.
    with torch.no_grad():
        generator = torch.Generator().manual_seed(7)
        for layer in model.model.layers:
            gate = getattr(layer.mlp, "gate", None)
            if gate is not None:
                gate.e_score_correction_bias.copy_(
                    torch.rand(gate.e_score_correction_bias.shape, generator=generator) * 0.2
                )
    return model
