import torch
from transformers import AutoModelForCausalLM

from winnow.adapters import adapter_for
from winnow.checkpoint import save_checkpoint
from winnow.collect import StatsCollector
from winnow.runtime import WinnowLagunaForCausalLM
from winnow.scoring import channel_scores, reap_scores
from winnow.selection import select_channels


def test_laguna_collection_and_channel_checkpoint(tmp_path, tiny_laguna, input_ids):
    adapter = adapter_for(tiny_laguna)
    assert adapter.family == "laguna"
    # Layer 0 is dense, so only the two sparse layers are adapted.
    assert [index for index, _block in adapter.layers] == [1, 2]
    with StatsCollector(tiny_laguna) as collector, torch.no_grad():
        tiny_laguna(input_ids, use_cache=False)
    assert channel_scores(collector.stats).shape == (2, 4, 8)
    assert reap_scores(collector.stats).shape == (2, 4)
    assert torch.equal(
        collector.stats["token_count"].sum(dim=1),
        torch.full((2,), input_ids.numel() * 2),
    )

    scores = torch.arange(64, dtype=torch.float32).reshape(2, 4, 8)
    plan = select_channels(scores, 0.5, top_k=2, layer_indices=(1, 2))
    save_checkpoint(tiny_laguna, plan, tmp_path, metadata={"test": True})
    loaded = AutoModelForCausalLM.from_pretrained(tmp_path, trust_remote_code=True).eval()
    assert isinstance(loaded, WinnowLagunaForCausalLM)

    source_shared = tiny_laguna.model.layers[1].mlp.shared_experts.down_proj.weight
    loaded_shared = loaded.model.layers[1].mlp.shared_experts.down_proj.weight
    torch.testing.assert_close(loaded_shared, source_shared)
    source_dense = tiny_laguna.model.layers[0].mlp.down_proj.weight
    loaded_dense = loaded.model.layers[0].mlp.down_proj.weight
    torch.testing.assert_close(loaded_dense, source_dense)
    for layer in (1, 2):
        gate = loaded.model.layers[layer].mlp.gate
        assert gate.e_score_correction_bias.shape[0] == gate.weight.shape[0]
    with torch.no_grad():
        logits = loaded(input_ids, use_cache=False).logits
    assert logits.shape == (2, 8, 64)
    assert torch.isfinite(logits).all()


def test_laguna_keep_one_checkpoint_matches_source(tmp_path, tiny_laguna, input_ids):
    scores = torch.ones(2, 4, 8)
    plan = select_channels(scores, 1.0, top_k=2, layer_indices=(1, 2))
    save_checkpoint(tiny_laguna, plan, tmp_path, metadata={"test": True})
    loaded = AutoModelForCausalLM.from_pretrained(tmp_path, trust_remote_code=True).eval()
    for layer in (1, 2):
        torch.testing.assert_close(
            loaded.model.layers[layer].mlp.gate.e_score_correction_bias,
            tiny_laguna.model.layers[layer].mlp.gate.e_score_correction_bias,
        )
    with torch.no_grad():
        expected = tiny_laguna(input_ids, use_cache=False).logits
        actual = loaded(input_ids, use_cache=False).logits
    torch.testing.assert_close(actual, expected)
