"""Command-line interface for Winnow."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch
import transformers
from huggingface_hub import HfApi
from huggingface_hub.errors import HfHubHTTPError
from transformers import AutoModelForCausalLM, AutoTokenizer

from . import __version__
from .adapters import adapter_for
from .calibration import calibration_batches
from .checkpoint import save_checkpoint
from .collect import StatsCollector
from .retention import validate_keep
from .scoring import channel_scores, reap_scores
from .selection import select_channels, select_experts


def _keep(value: str) -> float:
    try:
        return validate_keep(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def build_parser() -> argparse.ArgumentParser:
    """Build the public command parser."""
    parser = argparse.ArgumentParser(prog="winnow")
    subparsers = parser.add_subparsers(dest="command", required=True)
    prune = subparsers.add_parser("prune", help="score and prune a supported MoE model")
    prune.add_argument("model", help="Hugging Face model slug or local checkpoint")
    prune.add_argument("--keep", required=True, type=_keep, help="decimal fraction to keep")
    prune.add_argument("--calibration", required=True, help="Hugging Face dataset slug")
    prune.add_argument("--output", required=True, type=Path, help="output checkpoint path")
    prune.add_argument("--strategy", choices=("winnow", "reap"), default="winnow")
    prune.add_argument("--dataset-config")
    prune.add_argument("--split", default="train")
    prune.add_argument("--text-field", default="text")
    prune.add_argument("--model-revision")
    prune.add_argument("--dataset-revision")
    prune.add_argument("--sequences", type=int, default=128)
    prune.add_argument("--sequence-length", type=int, default=2048)
    prune.add_argument("--batch-size", type=int, default=1)
    prune.add_argument("--seed", type=int, default=0)
    prune.add_argument("--block-size", type=int, default=128)
    prune.add_argument("--device", default="auto")
    prune.add_argument(
        "--dtype",
        choices=("auto", "bfloat16", "float16", "float32"),
        default="auto",
    )
    prune.add_argument("--max-shard-size", default="5GB")
    return parser


def _device(value: str) -> str:
    if value != "auto":
        return value
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def _dtype(value: str, device: str) -> torch.dtype:
    if value == "auto":
        return torch.bfloat16 if device.startswith("cuda") else torch.float32
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[value]


def _hub_sha(kind: str, slug: str, revision: str | None) -> str | None:
    if Path(slug).exists():
        return None
    api = HfApi()
    try:
        if kind == "model":
            return api.model_info(slug, revision=revision).sha
        return api.dataset_info(slug, revision=revision).sha
    except (HfHubHTTPError, OSError, ValueError):
        return None


def _fingerprint(config) -> str:
    payload = json.dumps(config.to_dict(), sort_keys=True, default=str).encode()
    return hashlib.sha256(payload).hexdigest()


def run_prune(args: argparse.Namespace) -> Path:
    """Run one complete calibration and pruning job."""
    device = _device(args.device)
    dtype = _dtype(args.dtype, device)
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.model_revision)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        revision=args.model_revision,
        dtype=dtype,
        device_map={"": device},
    ).eval()
    adapter = adapter_for(model)

    batches = calibration_batches(
        tokenizer,
        args.calibration,
        dataset_config=args.dataset_config,
        split=args.split,
        text_field=args.text_field,
        revision=args.dataset_revision,
        sequences=args.sequences,
        sequence_length=args.sequence_length,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    with StatsCollector(model, adapter) as collector, torch.no_grad():
        for batch in batches:
            model(input_ids=batch.to(device), use_cache=False)

    layer_indices = tuple(index for index, _block in adapter.layers)
    if args.strategy == "winnow":
        plan = select_channels(
            channel_scores(collector.stats),
            args.keep,
            top_k=adapter.top_k,
            block_size=args.block_size,
            layer_indices=layer_indices,
        )
    else:
        plan = select_experts(
            reap_scores(collector.stats),
            args.keep,
            top_k=adapter.top_k,
            original_width=adapter.original_width,
            layer_indices=layer_indices,
        )

    metadata = {
        "winnow_version": __version__,
        "source_model": args.model,
        "source_revision": args.model_revision,
        "source_commit": _hub_sha("model", args.model, args.model_revision),
        "source_config_sha256": _fingerprint(model.config),
        "calibration": {
            "dataset": args.calibration,
            "config": args.dataset_config,
            "split": args.split,
            "text_field": args.text_field,
            "revision": args.dataset_revision,
            "commit": _hub_sha("dataset", args.calibration, args.dataset_revision),
            "sequences": args.sequences,
            "sequence_length": args.sequence_length,
            "tokens": collector.stats["n_tokens"],
            "batch_size": args.batch_size,
            "seed": args.seed,
            "packing": "EOS-separated contiguous blocks",
        },
        "score": (
            "mean(actual_router_weight * abs(post_swiglu_activation) * down_column_l2)"
            if args.strategy == "winnow"
            else "mean(normalized_topk_router_weight * ungated_expert_output_l2)"
        ),
        "dtype": str(dtype).removeprefix("torch."),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
    }
    return save_checkpoint(
        model,
        plan,
        args.output,
        metadata=metadata,
        tokenizer=tokenizer,
        max_shard_size=args.max_shard_size,
    )


def main(argv: list[str] | None = None) -> int:
    """Run the Winnow command."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        output = run_prune(args)
    except (RuntimeError, ValueError) as exc:
        parser.exit(2, f"winnow: error: {exc}\n")
    print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
