"""Command-line interface for Winnow."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
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


def _add_dataset_args(parser: argparse.ArgumentParser, *, sequences: int) -> None:
    parser.add_argument("--calibration", required=True, help="Hugging Face dataset slug")
    parser.add_argument("--dataset-config")
    parser.add_argument("--split", default="train")
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--dataset-revision")
    parser.add_argument("--sequences", type=int, default=sequences)
    parser.add_argument("--sequence-length", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=0)


def _add_prune_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--keep", required=True, type=_keep, help="decimal fraction to keep")
    parser.add_argument("--output", required=True, type=Path, help="output checkpoint path")
    parser.add_argument("--strategy", choices=("winnow", "reap"), default="winnow")
    parser.add_argument("--block-size", type=int, default=128)


def _add_stream_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--chunk-sequences", type=int, default=4)
    parser.add_argument("--ranks", type=int, default=0, help="0 = one per GPU")
    parser.add_argument("--workdir", type=Path, help="residual buffer directory")


def build_parser() -> argparse.ArgumentParser:
    """Build the public command parser."""
    parser = argparse.ArgumentParser(prog="winnow")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prune = subparsers.add_parser("prune", help="score and prune a supported MoE model")
    prune.add_argument("model", help="Hugging Face model slug or local checkpoint")
    _add_prune_args(prune)
    _add_dataset_args(prune, sequences=128)
    prune.add_argument("--model-revision")
    prune.add_argument("--batch-size", type=int, default=1)
    prune.add_argument("--device", default="auto")
    prune.add_argument(
        "--dtype",
        choices=("auto", "bfloat16", "float16", "float32"),
        default="auto",
    )
    prune.add_argument("--max-shard-size", default="5GB")
    prune.add_argument(
        "--max-gpu-memory",
        help=(
            "per-GPU memory cap for --device auto, e.g. 20GiB; leaves headroom "
            "for checkpoint-conversion scratch on tightly packed GPUs"
        ),
    )

    stream_prune = subparsers.add_parser(
        "stream-prune",
        help="score and prune a model too large to load, one layer at a time",
    )
    stream_prune.add_argument("model", help="Hugging Face model slug or local checkpoint")
    _add_prune_args(stream_prune)
    _add_dataset_args(stream_prune, sequences=1024)
    stream_prune.add_argument("--eval-sequences", type=int, default=64)
    _add_stream_args(stream_prune)

    stream_eval = subparsers.add_parser(
        "stream-eval",
        help="held-out perplexity of a checkpoint too large to load",
    )
    stream_eval.add_argument("model", help="model slug or (pruned) local checkpoint")
    _add_dataset_args(stream_eval, sequences=64)
    stream_eval.add_argument("--skip-sequences", type=int, default=0)
    _add_stream_args(stream_eval)
    return parser


def _device_map(value: str) -> str | dict[str, str]:
    if value == "auto":
        return "auto"
    return {"": value}


def _dtype(value: str, device: str) -> torch.dtype:
    if value == "auto":
        uses_cuda = device.startswith("cuda") or (device == "auto" and torch.cuda.is_available())
        return torch.bfloat16 if uses_cuda else torch.float32
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


_SCORE_FORMULAS = {
    "winnow": "mean(actual_router_weight * abs(post_swiglu_activation) * down_column_l2)",
    "reap": "mean(normalized_topk_router_weight * ungated_expert_output_l2)",
}


def _build_plan(stats, args: argparse.Namespace, *, top_k: int, original_width: int):
    """Select the pruning plan for one strategy from collected statistics."""
    if args.strategy == "winnow":
        return select_channels(
            channel_scores(stats),
            args.keep,
            top_k=top_k,
            block_size=args.block_size,
            layer_indices=stats["layer_indices"],
        )
    return select_experts(
        reap_scores(stats),
        args.keep,
        top_k=top_k,
        original_width=original_width,
        layer_indices=stats["layer_indices"],
    )


def _base_metadata(
    args: argparse.Namespace,
    *,
    config_fingerprint: str,
    tokens: int,
    revision: str | None = None,
    **calibration_extra,
) -> dict:
    """Provenance shared by both prune commands; callers add their own keys."""
    return {
        "winnow_version": __version__,
        "source_model": args.model,
        "source_commit": _hub_sha("model", args.model, revision),
        "source_config_sha256": config_fingerprint,
        "calibration": {
            "dataset": args.calibration,
            "config": args.dataset_config,
            "split": args.split,
            "text_field": args.text_field,
            "revision": args.dataset_revision,
            "commit": _hub_sha("dataset", args.calibration, args.dataset_revision),
            "sequences": args.sequences,
            "sequence_length": args.sequence_length,
            "tokens": tokens,
            "seed": args.seed,
            "packing": "EOS-separated contiguous blocks",
            **calibration_extra,
        },
        "score": _SCORE_FORMULAS[args.strategy],
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
    }


def run_prune(args: argparse.Namespace) -> Path:
    """Run one complete calibration and pruning job."""
    dtype = _dtype(args.dtype, args.device)
    tokenizer = _load_tokenizer(args.model, revision=args.model_revision)
    load_kwargs = {}
    if args.max_gpu_memory and args.device == "auto":
        load_kwargs["max_memory"] = {
            index: args.max_gpu_memory for index in range(torch.cuda.device_count())
        }
        load_kwargs["max_memory"]["cpu"] = "64GiB"
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        revision=args.model_revision,
        dtype=dtype,
        device_map=_device_map(args.device),
        **load_kwargs,
    ).eval()
    adapter = adapter_for(model)
    if any(parameter.is_meta for parameter in model.parameters()):
        raise RuntimeError(
            "the model does not fit on the available devices (some weights are "
            "CPU-offloaded, which the in-memory checkpoint writer cannot read); "
            "use `winnow stream-prune` for models this large"
        )
    input_device = model.get_input_embeddings().weight.device

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
            model(input_ids=batch.to(input_device), use_cache=False)

    plan = _build_plan(
        collector.stats, args, top_k=adapter.top_k, original_width=adapter.original_width
    )
    metadata = _base_metadata(
        args,
        config_fingerprint=_fingerprint(model.config),
        tokens=collector.stats["n_tokens"],
        revision=args.model_revision,
        batch_size=args.batch_size,
    )
    metadata.update(
        {
            "source_revision": args.model_revision,
            "dtype": str(dtype).removeprefix("torch."),
            "device_map": {
                key: str(value) for key, value in getattr(model, "hf_device_map", {}).items()
            },
        }
    )
    return save_checkpoint(
        model,
        plan,
        args.output,
        metadata=metadata,
        tokenizer=tokenizer,
        max_shard_size=args.max_shard_size,
    )


def _load_tokenizer(checkpoint, revision: str | None = None):
    """Load a tokenizer, applying the Mistral regex fix for Laguna models."""
    from .stream import load_config

    config = load_config(checkpoint)
    kwargs = {}
    if str(getattr(config, "model_type", "")).startswith("winnow_"):
        # Our own checkpoints carry an auto_map to the winnow shim modules;
        # AutoTokenizer consults it, so trust it to avoid the interactive
        # prompt (the "custom code" is winnow's own installed package).
        kwargs["trust_remote_code"] = True
    if "laguna" in str(getattr(config, "model_type", "")):
        # Laguna ships a Mistral-family tokenizer with the known faulty
        # pretokenizer regex; transformers warns that unfixed tokenization is
        # incorrect, so opt in to the fix.
        kwargs["fix_mistral_regex"] = True
    return AutoTokenizer.from_pretrained(checkpoint, revision=revision, **kwargs)


def _resolve_checkpoint(slug_or_path: str, revision: str | None = None) -> Path:
    """Return a local snapshot directory for a slug or an existing path."""
    path = Path(slug_or_path)
    if path.exists():
        return path
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(slug_or_path, revision=revision))


def _collect_sequences(tokenizer, args: argparse.Namespace, total: int, skip: int = 0):
    rows = []
    batches = calibration_batches(
        tokenizer,
        args.calibration,
        dataset_config=args.dataset_config,
        split=args.split,
        text_field=args.text_field,
        revision=args.dataset_revision,
        sequences=skip + total,
        sequence_length=args.sequence_length,
        batch_size=1,
        seed=args.seed,
    )
    for batch in batches:
        rows.append(batch[0])
    return torch.stack(rows[skip:])


def _stream_ranks(requested: int) -> int:
    if requested > 0:
        return requested
    return max(torch.cuda.device_count(), 1)


def run_stream_prune(args: argparse.Namespace) -> Path:
    """Score, prune, and write a checkpoint without ever loading the model."""
    from .checkpoint import save_checkpoint_streamed
    from .stream import load_config, stream_run

    checkpoint = _resolve_checkpoint(args.model)
    config = load_config(checkpoint)
    tokenizer = _load_tokenizer(checkpoint)
    sequences = _collect_sequences(tokenizer, args, args.sequences + args.eval_sequences)
    ranks = _stream_ranks(args.ranks)
    workdir = args.workdir or args.output.parent / f"{args.output.name}-streamwork"

    result = stream_run(
        checkpoint,
        sequences,
        calibration_sequences=args.sequences,
        workdir=workdir,
        ranks=ranks,
        chunk_sequences=args.chunk_sequences,
    )
    stats = result["stats"]
    torch.save(stats, Path(workdir) / "stats.pt")
    print(f"source perplexity {result['perplexity']:.4f} on {args.eval_sequences} sequences")

    plan = _build_plan(
        stats,
        args,
        top_k=config.num_experts_per_tok,
        original_width=config.moe_intermediate_size,
    )
    metadata = _base_metadata(
        args,
        config_fingerprint=_fingerprint(config),
        tokens=stats["n_tokens"],
        eval_sequences=args.eval_sequences,
    )
    metadata.update(
        {
            "source_perplexity": result["perplexity"],
            "pipeline": f"layer-streamed, {ranks} data-parallel ranks",
        }
    )
    return save_checkpoint_streamed(
        checkpoint, plan, args.output, metadata=metadata, tokenizer=tokenizer
    )


def run_stream_eval(args: argparse.Namespace) -> float:
    """Held-out perplexity of a (possibly pruned) checkpoint, layer-streamed."""
    from .stream import stream_run

    checkpoint = _resolve_checkpoint(args.model)
    tokenizer = _load_tokenizer(checkpoint)
    sequences = _collect_sequences(tokenizer, args, args.sequences, skip=args.skip_sequences)
    # Not under `checkpoint`: for a hub slug that is the read-only cache snapshot.
    workdir = args.workdir or Path(tempfile.mkdtemp(prefix="winnow-streameval-"))
    result = stream_run(
        checkpoint,
        sequences,
        calibration_sequences=0,
        workdir=workdir,
        ranks=_stream_ranks(args.ranks),
        collect_stats=False,
        chunk_sequences=args.chunk_sequences,
    )
    print(
        f"perplexity {result['perplexity']:.4f} "
        f"({result['loss_tokens']} scored tokens, {args.sequences} sequences)"
    )
    return result["perplexity"]


def main(argv: list[str] | None = None) -> int:
    """Run the Winnow command."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "prune":
            output = run_prune(args)
            print(f"Wrote {output}")
        elif args.command == "stream-prune":
            output = run_stream_prune(args)
            print(f"Wrote {output}")
        else:
            run_stream_eval(args)
    except (RuntimeError, ValueError) as exc:
        parser.exit(2, f"winnow: error: {exc}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
