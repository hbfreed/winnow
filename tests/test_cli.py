import pytest
import torch

from winnow.cli import _device_map, _dtype, build_parser


def test_minimum_prune_command():
    args = build_parser().parse_args(
        [
            "prune",
            "allenai/OLMoE-1B-7B-0924",
            "--keep",
            "0.5",
            "--calibration",
            "allenai/c4",
            "--output",
            "out",
        ]
    )
    assert args.keep == 0.5
    assert args.strategy == "winnow"


def test_cli_rejects_percentage():
    with pytest.raises(SystemExit):
        build_parser().parse_args(
            [
                "prune",
                "model",
                "--keep",
                "50",
                "--calibration",
                "dataset",
                "--output",
                "out",
            ]
        )


def test_auto_device_uses_accelerate_and_cuda_dtype(monkeypatch):
    assert _device_map("auto") == "auto"
    assert _device_map("cpu") == {"": "cpu"}
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert _dtype("auto", "auto") == torch.bfloat16
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert _dtype("auto", "auto") == torch.float32
