"""Stream and pack calibration text."""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

import torch
from datasets import load_dataset

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase


def calibration_batches(
    tokenizer: PreTrainedTokenizerBase,
    dataset: str,
    *,
    dataset_config: str | None = None,
    split: str = "train",
    text_field: str = "text",
    revision: str | None = None,
    sequences: int = 128,
    sequence_length: int = 2048,
    batch_size: int = 1,
    seed: int = 0,
) -> Iterator[torch.LongTensor]:
    """Yield deterministic, EOS-separated token blocks from a Hub dataset."""
    if sequences <= 0:
        raise ValueError("sequences must be greater than 0")
    if sequence_length <= 0:
        raise ValueError("sequence_length must be greater than 0")
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than 0")
    if tokenizer.eos_token_id is None:
        raise ValueError("the tokenizer must define eos_token_id")

    kwargs: dict[str, Any] = {
        "path": dataset,
        "split": split,
        "streaming": True,
    }
    if dataset_config is not None:
        kwargs["name"] = dataset_config
    if revision is not None:
        kwargs["revision"] = revision
    stream = load_dataset(**kwargs)
    if hasattr(stream, "select_columns"):
        stream = stream.select_columns([text_field])
    stream = stream.shuffle(seed=seed, buffer_size=10_000)

    token_buffer: list[int] = []
    rows: list[torch.Tensor] = []
    produced = 0
    eos = int(tokenizer.eos_token_id)
    for example in stream:
        if text_field not in example:
            raise ValueError(f"dataset rows do not contain the field {text_field!r}")
        token_ids = tokenizer(str(example[text_field]), add_special_tokens=False)["input_ids"]
        if not token_ids:
            continue
        token_buffer.extend(token_ids)
        token_buffer.append(eos)
        while len(token_buffer) >= sequence_length and produced < sequences:
            rows.append(torch.tensor(token_buffer[:sequence_length], dtype=torch.long))
            del token_buffer[:sequence_length]
            produced += 1
            if len(rows) == batch_size or produced == sequences:
                yield torch.stack(rows)
                rows = []
        if produced == sequences:
            return
    raise RuntimeError(
        f"the calibration dataset produced {produced} of {sequences} requested sequences"
    )
