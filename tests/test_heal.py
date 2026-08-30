import torch
from transformers import AutoModelForCausalLM

from winnow.checkpoint import save_checkpoint
from winnow.heal import attach_healing, merge_healing
from winnow.selection import select_channels


def _pruned(tmp_path, tiny_afmoe):
    scores = torch.arange(64, dtype=torch.float32).reshape(2, 4, 8)
    plan = select_channels(scores, 0.5, top_k=2, layer_indices=(1, 2))
    save_checkpoint(tiny_afmoe, plan, tmp_path, metadata={"test": True})
    return AutoModelForCausalLM.from_pretrained(tmp_path, trust_remote_code=True).eval()


def test_fresh_adapters_do_not_change_outputs(tmp_path, tiny_afmoe, input_ids):
    model = _pruned(tmp_path, tiny_afmoe)
    with torch.no_grad():
        before = model(input_ids, use_cache=False).logits
    attach_healing(model, rank=2, alpha=4.0)
    with torch.no_grad():
        after = model(input_ids, use_cache=False).logits
    torch.testing.assert_close(after, before)


def test_only_healing_parameters_train(tmp_path, tiny_afmoe, input_ids):
    model = _pruned(tmp_path, tiny_afmoe)
    trainable = attach_healing(model, rank=2, alpha=4.0)
    assert any(".lora_gate_up_a." in name for name in trainable)
    assert any(".self_attn.q_proj.lora_a" in name for name in trainable)
    assert any("router.gate.weight" in name for name in trainable)
    assert not any(".gate_up_projs." in name for name in trainable)
    assert not any("embed_tokens" in name for name in trainable)

    loss = model(input_ids, use_cache=False).logits.float().pow(2).mean()
    loss.backward()
    grads = [
        name
        for name, parameter in model.named_parameters()
        if parameter.grad is not None and parameter.grad.abs().sum() > 0
    ]
    assert any(".lora_gate_up_b." in name for name in grads)
    assert any("router.gate.weight" in name for name in grads)
    assert all(name in trainable for name in grads)


def test_merge_matches_adapted_forward(tmp_path, tiny_afmoe, input_ids):
    model = _pruned(tmp_path, tiny_afmoe)
    attach_healing(model, rank=2, alpha=4.0)
    # Give the adapters real values so the merge is not a zero no-op.
    generator = torch.Generator().manual_seed(3)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if "lora_" in name:
                parameter.copy_(torch.randn(parameter.shape, generator=generator) * 0.05)
    with torch.no_grad():
        adapted = model(input_ids, use_cache=False).logits
    merge_healing(model)
    assert not any("lora_" in name for name, _p in model.named_parameters())
    with torch.no_grad():
        merged = model(input_ids, use_cache=False).logits
    torch.testing.assert_close(merged, adapted, rtol=1e-4, atol=1e-5)


def test_merged_model_roundtrips_through_save(tmp_path, tiny_afmoe, input_ids):
    model = _pruned(tmp_path / "pruned", tiny_afmoe)
    attach_healing(model, rank=2, alpha=4.0)
    merge_healing(model)
    with torch.no_grad():
        expected = model(input_ids, use_cache=False).logits
    model.save_pretrained(tmp_path / "healed")
    reloaded = AutoModelForCausalLM.from_pretrained(
        tmp_path / "healed", trust_remote_code=True
    ).eval()
    with torch.no_grad():
        actual = reloaded(input_ids, use_cache=False).logits
    torch.testing.assert_close(actual, expected)
