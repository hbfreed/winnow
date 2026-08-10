import pytest

from winnow.cli import build_parser


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
