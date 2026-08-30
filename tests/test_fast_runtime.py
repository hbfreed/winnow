import importlib.util

import pytest
import torch
import torch.nn.functional as F

pytest.importorskip("megablocks.backend.fused_moe")

from winnow.runtime.fast import FastOlmoeMoE, FastQwenMoE


def test_fast_runtime_packs_checkpoint_weights():
    module = FastOlmoeMoE(128, [128, 256], 1)
    generator = torch.Generator().manual_seed(3)
    gate = torch.randn(128, 128, generator=generator)
    up = torch.randn(128, 128, generator=generator)
    down = torch.randn(128, 128, generator=generator)
    module.load_expert_weight_(0, "gate", gate)
    module.load_expert_weight_(0, "up", up)
    module.load_expert_weight_(0, "down", down)
    torch.testing.assert_close(module.expert_weight(0, "gate"), gate)
    torch.testing.assert_close(module.expert_weight(0, "up"), up)
    torch.testing.assert_close(module.expert_weight(0, "down"), down)


def test_fast_runtime_rejects_unaligned_widths():
    with pytest.raises(ValueError, match="multiples of 128"):
        FastQwenMoE(128, [64, 128], 1)


def test_fork_install_does_not_build_training_extensions():
    assert importlib.util.find_spec("megablocks_ops") is None
    assert importlib.util.find_spec("nanomoe_ops") is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_fast_olmoe_matches_reference_math():
    torch.manual_seed(7)
    device = torch.device("cuda")
    module = FastOlmoeMoE(128, [128, 256, 128], 2).to(device=device, dtype=torch.bfloat16)
    with torch.no_grad():
        module.gate.weight.normal_(std=0.02)
        module.w_gate.normal_(std=0.02)
        module.w_up.normal_(std=0.02)
        module.w_down.normal_(std=0.02)
        inputs = torch.randn(11, 128, device=device, dtype=torch.bfloat16)
        actual = module(inputs)

        probabilities = F.softmax(module.gate(inputs), dim=-1, dtype=torch.float32)
        router_weights, experts = torch.topk(probabilities, 2, dim=-1)
        expected = torch.zeros_like(inputs, dtype=torch.float32)
        for expert in range(module.num_experts):
            tokens, slots = torch.where(experts == expert)
            if tokens.numel() == 0:
                continue
            current = inputs[tokens].float()
            gate = F.linear(current, module.expert_weight(expert, "gate").float())
            up = F.linear(current, module.expert_weight(expert, "up").float())
            current = F.silu(gate) * up
            current = F.linear(current, module.expert_weight(expert, "down").float())
            current *= router_weights[tokens, slots, None]
            expected.index_add_(0, tokens, current)

    relative_error = (actual.float() - expected).norm() / expected.norm()
    assert relative_error < 5e-3


def test_quantize_at_load_matches_post_hoc_quantization():
    from winnow.runtime.fast import FastSigmoidMoE

    generator = torch.Generator().manual_seed(21)
    widths = [128, 256]
    slabs = {
        (expert, kind): torch.randn(shape, generator=generator, dtype=torch.bfloat16)
        for expert, width in enumerate(widths)
        for kind, shape in (
            ("gate", (width, 128)),
            ("up", (width, 128)),
            ("down", (128, width)),
        )
    }
    post_hoc = FastSigmoidMoE(128, widths, 1, dtype=torch.bfloat16)
    at_load = FastSigmoidMoE(128, widths, 1, quantize_w8a16=True)
    assert at_load.is_int8  # int8 buffers exist before any weight loads
    for (expert, kind), weight in slabs.items():
        post_hoc.load_expert_weight_(expert, kind, weight)
        at_load.load_expert_weight_(expert, kind, weight)
    post_hoc.quantize_int8_()
    for name in ("w_gate", "w_up", "w_down", "w_gate_scale", "w_up_scale", "w_down_scale"):
        torch.testing.assert_close(
            getattr(at_load, name), getattr(post_hoc, name), rtol=0, atol=0
        )
