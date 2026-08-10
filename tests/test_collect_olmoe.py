import torch
import torch.nn.functional as F

from winnow.collect import StatsCollector


def test_collector_uses_actual_channels_and_normalized_reap(tiny_olmoe, input_ids):
    model = tiny_olmoe
    block = model.model.layers[0].mlp
    with torch.no_grad():
        block.gate.weight.zero_()

    captured = {}

    def capture(_module, args, _kwargs):
        captured["hidden"] = args[0].detach().clone()
        captured["indices"] = args[1].detach().clone()

    handle = block.experts.register_forward_pre_hook(capture, with_kwargs=True)
    try:
        with StatsCollector(model) as collector, torch.no_grad():
            model(input_ids, use_cache=False)
    finally:
        handle.remove()

    hidden = captured["hidden"]
    indices = captured["indices"]
    expected_count = torch.zeros(4, dtype=torch.int64)
    expected_reap = torch.zeros(4)
    expected_channels = torch.zeros(4, 8)
    for token in range(hidden.shape[0]):
        for slot in range(indices.shape[1]):
            expert = int(indices[token, slot])
            gate, up = F.linear(
                hidden[token : token + 1], block.experts.gate_up_proj[expert]
            ).chunk(2, dim=-1)
            activation = block.experts.act_fn(gate) * up
            output = F.linear(activation, block.experts.down_proj[expert])
            expected_count[expert] += 1
            expected_reap[expert] += 0.5 * output.float().norm()
            expected_channels[expert] += 0.25 * activation.abs().float().squeeze(0)

    assert torch.equal(collector.stats["token_count"][0], expected_count)
    torch.testing.assert_close(collector.stats["reap_sum"][0], expected_reap)
    torch.testing.assert_close(collector.stats["channel_sum"][0], expected_channels)
    torch.testing.assert_close(
        collector.stats["down_norm"][0],
        block.experts.down_proj.detach().float().norm(dim=1),
    )


def test_collector_accumulates_batches_and_drops_padding(tiny_olmoe, input_ids):
    mask = torch.ones_like(input_ids)
    mask[0, -2:] = 0
    with StatsCollector(tiny_olmoe) as collector, torch.no_grad():
        tiny_olmoe(input_ids, attention_mask=mask, use_cache=False)
        tiny_olmoe(input_ids[:1], use_cache=False)
    expected_tokens = int(mask.sum()) + input_ids.shape[1]
    assert collector.stats["n_tokens"] == expected_tokens
    expected_routes = expected_tokens * tiny_olmoe.config.num_experts_per_tok
    assert torch.equal(
        collector.stats["token_count"].sum(dim=1),
        torch.full((2,), expected_routes),
    )
