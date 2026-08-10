import torch
from transformers import AutoModelForCausalLM

from winnow.adapters import adapter_for
from winnow.checkpoint import save_checkpoint
from winnow.collect import StatsCollector
from winnow.runtime import WinnowQwen3_5MoeForCausalLM
from winnow.scoring import channel_scores, reap_scores
from winnow.selection import select_channels


def test_qwen_collection_and_channel_checkpoint(tmp_path, tiny_qwen, input_ids):
    adapter = adapter_for(tiny_qwen)
    assert adapter.family == "qwen3_5_moe"
    with StatsCollector(tiny_qwen) as collector, torch.no_grad():
        tiny_qwen(input_ids, use_cache=False)
    assert channel_scores(collector.stats).shape == (2, 4, 8)
    assert reap_scores(collector.stats).shape == (2, 4)
    assert torch.equal(
        collector.stats["token_count"].sum(dim=1),
        torch.full((2,), input_ids.numel() * 2),
    )

    scores = torch.arange(64, dtype=torch.float32).reshape(2, 4, 8)
    plan = select_channels(scores, 0.5, top_k=2)
    save_checkpoint(tiny_qwen, plan, tmp_path, metadata={"test": True})
    loaded = AutoModelForCausalLM.from_pretrained(tmp_path, trust_remote_code=True).eval()
    assert isinstance(loaded, WinnowQwen3_5MoeForCausalLM)

    source_shared = tiny_qwen.model.layers[0].mlp.shared_expert.down_proj.weight
    loaded_shared = loaded.model.layers[0].mlp.shared_expert.down_proj.weight
    torch.testing.assert_close(loaded_shared, source_shared)
    with torch.no_grad():
        logits = loaded(input_ids, use_cache=False).logits
    assert logits.shape == (2, 8, 64)
    assert torch.isfinite(logits).all()


def test_qwen_keep_one_checkpoint_matches_source(tmp_path, tiny_qwen, input_ids):
    scores = torch.ones(2, 4, 8)
    plan = select_channels(scores, 1.0, top_k=2)
    save_checkpoint(tiny_qwen, plan, tmp_path, metadata={"test": True})
    loaded = AutoModelForCausalLM.from_pretrained(tmp_path, trust_remote_code=True).eval()
    with torch.no_grad():
        expected = tiny_qwen(input_ids, use_cache=False).logits
        actual = loaded(input_ids, use_cache=False).logits
    torch.testing.assert_close(actual, expected)
