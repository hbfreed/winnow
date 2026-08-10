import json

import torch
from transformers import AutoModelForCausalLM

from winnow.checkpoint import extract_state, save_checkpoint
from winnow.runtime import WinnowOlmoeForCausalLM
from winnow.selection import select_channels, select_experts


def test_channel_state_uses_exact_rows_and_columns(tiny_olmoe):
    scores = torch.arange(64, dtype=torch.float32).reshape(2, 4, 8)
    plan = select_channels(scores, 0.5, top_k=2)
    state = extract_state(tiny_olmoe, plan)
    for layer_plan in plan.layers:
        source = tiny_olmoe.model.layers[layer_plan.layer].mlp
        expected_router = source.gate.weight[list(layer_plan.experts)]
        torch.testing.assert_close(
            state[f"model.layers.{layer_plan.layer}.mlp.gate.weight"], expected_router
        )
        for slot, (expert, channels) in enumerate(
            zip(layer_plan.experts, layer_plan.channels, strict=True)
        ):
            channel_tensor = torch.tensor(channels)
            rows = torch.cat([channel_tensor, channel_tensor + 8])
            torch.testing.assert_close(
                state[f"model.layers.{layer_plan.layer}.mlp.experts.gate_up_projs.{slot}"],
                source.experts.gate_up_proj[expert, rows],
            )
            torch.testing.assert_close(
                state[f"model.layers.{layer_plan.layer}.mlp.experts.down_projs.{slot}"],
                source.experts.down_proj[expert, :, channel_tensor],
            )


def test_reap_selection_surgery_and_auto_load(tmp_path, tiny_olmoe, input_ids):
    scores = torch.tensor([[1.0, 4.0, 3.0, 2.0], [4.0, 1.0, 2.0, 3.0]])
    plan = select_experts(scores, 0.5, top_k=2, original_width=8)
    assert plan.layers[0].experts == (1, 2)
    assert plan.layers[1].experts == (0, 3)

    save_checkpoint(tiny_olmoe, plan, tmp_path, metadata={"test": True})
    metadata = json.loads((tmp_path / "winnow.json").read_text())
    assert metadata["schema_version"] == 1
    assert metadata["plan"]["strategy"] == "reap"

    loaded = AutoModelForCausalLM.from_pretrained(tmp_path, trust_remote_code=True).eval()
    assert isinstance(loaded, WinnowOlmoeForCausalLM)
    for position, survivors in enumerate(((1, 2), (0, 3))):
        source_gate = tiny_olmoe.model.layers[position].mlp.gate.weight
        torch.testing.assert_close(
            loaded.model.layers[position].mlp.gate.weight,
            source_gate[list(survivors)],
        )
    with torch.no_grad():
        logits = loaded(input_ids, use_cache=False).logits
    assert logits.shape == (2, 8, 64)
    assert torch.isfinite(logits).all()


def test_keep_one_checkpoint_matches_source(tmp_path, tiny_olmoe, input_ids):
    scores = torch.ones(2, 4, 8)
    plan = select_channels(scores, 1.0, top_k=2)
    save_checkpoint(tiny_olmoe, plan, tmp_path, metadata={"test": True})
    loaded = AutoModelForCausalLM.from_pretrained(tmp_path, trust_remote_code=True).eval()
    with torch.no_grad():
        expected = tiny_olmoe(input_ids, use_cache=False).logits
        actual = loaded(input_ids, use_cache=False).logits
    torch.testing.assert_close(actual, expected)
