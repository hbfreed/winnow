"""Contracts for the packed INT8 W8A16 quantization and the sigmoid fast block."""

import pytest
import torch
import torch.nn.functional as F

pytest.importorskip("megablocks.backend.fused_moe")

from winnow.runtime.fast import FastOlmoeMoE, FastSigmoidMoE


def _filled(module, seed: int = 5):
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.copy_(torch.randn(parameter.shape, generator=generator) * 0.02)
    return module


def test_quantize_int8_shapes_and_idempotence():
    module = _filled(FastOlmoeMoE(128, [128, 256], 1))
    reference = module.expert_weight(0, "gate").clone()
    module.quantize_int8_()
    assert module.is_int8
    assert module.w_gate.dtype == torch.int8
    assert module.w_gate_scale.shape == (384,)
    assert module.w_down_scale.shape == (2, 128)
    module.quantize_int8_()  # idempotent, must not double-quantize

    dequantized = module.w_gate.float() * module.w_gate_scale
    width = module.expert_widths[0]
    # Rounding error is bounded by half of one per-column scale step.
    max_step = module.w_gate_scale[:width].max()
    assert (dequantized[:, :width] - reference.t()).abs().max() <= max_step * 0.5 + 1e-8


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_fast_sigmoid_int8_matches_bf16():
    torch.manual_seed(9)
    device = torch.device("cuda")
    module = FastSigmoidMoE(128, [128, 256, 128], 2, routed_scaling_factor=2.5)
    module.shared_experts = torch.nn.Linear(128, 128, bias=False)
    _filled(module)
    with torch.no_grad():
        module.e_score_correction_bias.copy_(torch.rand(3) * 0.2)
    module = module.to(device=device, dtype=torch.bfloat16)

    inputs = torch.randn(37, 128, device=device, dtype=torch.bfloat16)
    with torch.no_grad():
        expected = module(inputs)
        module.quantize_int8_()
        actual = module(inputs)

    relative_error = (actual.float() - expected.float()).norm() / expected.float().norm()
    assert relative_error < 0.02


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_fast_sigmoid_matches_reference_math():
    torch.manual_seed(11)
    device = torch.device("cuda")
    module = FastSigmoidMoE(128, [128, 128], 2, routed_scaling_factor=2.5)
    module.shared_experts = torch.nn.Linear(128, 128, bias=False)
    _filled(module)
    with torch.no_grad():
        module.e_score_correction_bias.copy_(torch.rand(2) * 0.2)
    module = module.to(device=device, dtype=torch.bfloat16)

    inputs = torch.randn(13, 128, device=device, dtype=torch.bfloat16)
    with torch.no_grad():
        actual = module(inputs)

        scores = torch.sigmoid(module.gate(inputs).float())
        _, experts = torch.topk(scores + module.e_score_correction_bias.float(), 2, dim=-1)
        weights = scores.gather(-1, experts)
        weights = weights / weights.sum(dim=-1, keepdim=True)
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
            current *= weights[tokens, slots, None]
            expected.index_add_(0, tokens, current)
        expected = expected * 2.5 + module.shared_experts(inputs).float()

    relative_error = (actual.float() - expected).norm() / expected.norm()
    assert relative_error < 5e-3
