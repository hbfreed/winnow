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
    chat_template: bool = False,
) -> Iterator[torch.LongTensor]:
    """Yield deterministic, EOS-separated token blocks from a Hub dataset.

    With ``chat_template`` the ``text_field`` holds a conversation (a list of
    ``{"role", "content"}`` messages) that is rendered through the tokenizer's
    chat template before tokenization, so calibration matches how a post-trained
    model sees its deployment inputs.
    """
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

    def _render(values: list) -> list[str]:
        if not chat_template:
            return [str(value) for value in values]
        # The template supplies its own special tokens, so the rendered text
        # goes through the same add_special_tokens=False tokenization below.
        return tokenizer.apply_chat_template(values, tokenize=False)

    def _encoded() -> Iterator[list[int]]:
        # Fast tokenizers only parallelize across a batch; per-document calls
        # would keep the whole calibration set single-core.
        values: list = []
        for example in stream:
            if text_field not in example:
                raise ValueError(f"dataset rows do not contain the field {text_field!r}")
            values.append(example[text_field])
            if len(values) == 64:
                yield from tokenizer(_render(values), add_special_tokens=False)["input_ids"]
                values = []
        if values:
            yield from tokenizer(_render(values), add_special_tokens=False)["input_ids"]

    token_buffer: list[int] = []
    rows: list[torch.Tensor] = []
    produced = 0
    eos = int(tokenizer.eos_token_id)
    for token_ids in _encoded():
        if not token_ids:
            continue
        token_buffer.extend(token_ids)
        # A chat template usually closes the conversation with EOS already;
        # avoid stacking a second separator in that case.
        if token_ids[-1] != eos:
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
