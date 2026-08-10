import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

import pytest
import torch
from transformers import OlmoeConfig, OlmoeForCausalLM
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
