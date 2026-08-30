"""Write extra keep-level checkpoints from an existing stream-prune stats.pt.

Reuses the calibration statistics of the completed keep-0.5 Trinity-Mini run,
so no GPU sweep is needed. Run from the repo root after the main run finishes.
"""

import sys
from argparse import Namespace
from pathlib import Path

import torch

from winnow.checkpoint import save_checkpoint_streamed
from winnow.cli import _base_metadata, _build_plan, _fingerprint, _load_tokenizer, _resolve_checkpoint
from winnow.stream import load_config

MODEL = "arcee-ai/Trinity-Mini"
WORKDIR = Path("trinity-mini-winnow-50-streamwork")
SOURCE_PPL = float(sys.argv[1]) if len(sys.argv) > 1 else None
KEEPS = [float(value) for value in sys.argv[2:]] or [0.75, 0.25]

stats = torch.load(WORKDIR / "stats.pt", weights_only=False)
checkpoint = _resolve_checkpoint(MODEL)
config = load_config(checkpoint)
tokenizer = _load_tokenizer(checkpoint)

for keep in KEEPS:
    args = Namespace(
        model=MODEL,
        keep=keep,
        strategy="winnow",
        block_size=128,
        calibration="allenai/tulu-3-sft-mixture",
        dataset_config=None,
        split="train",
        text_field="messages",
        chat_template=True,
        dataset_revision=None,
        sequences=1024,
        sequence_length=2048,
        seed=0,
    )
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
        eval_sequences=64,
    )
    metadata.update(
        {
            "source_perplexity": SOURCE_PPL,
            "pipeline": "layer-streamed, 2 data-parallel ranks (stats reused from the keep-0.5 run)",
        }
    )
    output = Path(f"trinity-mini-winnow-{int(keep * 100)}")
    save_checkpoint_streamed(checkpoint, plan, output, metadata=metadata, tokenizer=tokenizer)
    print(f"Wrote {output}")
