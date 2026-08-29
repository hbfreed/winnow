import math

import pytest
import torch
import torch.nn.functional as F

from winnow.checkpoint import save_checkpoint
from winnow.collect import StatsCollector
from winnow.scoring import channel_scores, reap_scores
from winnow.selection import select_channels
from winnow.stream import stream_run


def _sequences(count: int = 6, length: int = 8) -> torch.Tensor:
    generator = torch.Generator().manual_seed(11)
    return torch.randint(0, 64, (count, length), generator=generator)


def _direct_perplexity(model, sequences: torch.Tensor) -> float:
    with torch.no_grad():
        logits = model(sequences, use_cache=False).logits.float()
    loss = F.cross_entropy(
        logits[:, :-1].flatten(0, 1), sequences[:, 1:].flatten(), reduction="mean"
    )
    return math.exp(float(loss))


@pytest.fixture(params=["tiny_laguna", "tiny_afmoe"])
def tiny_streamable(request):
    return request.getfixturevalue(request.param)


def test_stream_matches_full_forward(tmp_path, tiny_streamable):
    model = tiny_streamable
    sequences = _sequences()
    calibration = sequences[:4]
    held_out = sequences[4:]
    source_dir = tmp_path / "src"
    model.save_pretrained(source_dir)

    with StatsCollector(model) as collector, torch.no_grad():
        model(calibration, use_cache=False)
    expected_ppl = _direct_perplexity(model, held_out)

    result = stream_run(
        source_dir,
        sequences,
        calibration_sequences=4,
        workdir=tmp_path / "work",
        ranks=1,
        device="cpu",
        dtype=torch.float32,
        chunk_sequences=3,
    )

    assert result["stats"]["n_tokens"] == calibration.numel()
    torch.testing.assert_close(
        channel_scores(result["stats"]),
        channel_scores(collector.stats),
        rtol=1e-4,
        atol=1e-5,
    )
    torch.testing.assert_close(
        reap_scores(result["stats"]),
        reap_scores(collector.stats),
        rtol=1e-4,
        atol=1e-5,
    )
    assert result["perplexity"] == pytest.approx(expected_ppl, rel=1e-3)


def test_stream_pruned_checkpoint_perplexity(tmp_path, tiny_streamable):
    model = tiny_streamable
    sequences = _sequences()
    source_dir = tmp_path / "src"
    model.save_pretrained(source_dir)

    scores = torch.ones(2, 4, 8)
    plan = select_channels(scores, 1.0, top_k=2, layer_indices=(1, 2))
    pruned_dir = tmp_path / "pruned"
    save_checkpoint(model, plan, pruned_dir, metadata={"test": True})

    kwargs = {
        "calibration_sequences": 0,
        "ranks": 1,
        "device": "cpu",
        "dtype": torch.float32,
        "collect_stats": False,
    }
    base = stream_run(source_dir, sequences, workdir=tmp_path / "work_base", **kwargs)
    pruned = stream_run(pruned_dir, sequences, workdir=tmp_path / "work_pruned", **kwargs)
    assert pruned["perplexity"] == pytest.approx(base["perplexity"], rel=1e-4)


def test_streamed_writer_matches_in_memory_writer(tmp_path, tiny_streamable, input_ids):
    from transformers import AutoModelForCausalLM

    from winnow.checkpoint import save_checkpoint_streamed
    from winnow.stream import ShardReader

    model = tiny_streamable
    source_dir = tmp_path / "src"
    model.save_pretrained(source_dir)

    scores = torch.arange(64, dtype=torch.float32).reshape(2, 4, 8)
    plan = select_channels(scores, 0.5, top_k=2, layer_indices=(1, 2))
    memory_dir = tmp_path / "memory"
    streamed_dir = tmp_path / "streamed"
    save_checkpoint(model, plan, memory_dir, metadata={"test": True})
    save_checkpoint_streamed(source_dir, plan, streamed_dir, metadata={"test": True})

    memory = ShardReader(memory_dir)
    streamed = ShardReader(streamed_dir)
    assert sorted(memory.names) == sorted(streamed.names)
    for name in memory.names:
        torch.testing.assert_close(streamed.get(name), memory.get(name), rtol=0, atol=0)

    loaded = AutoModelForCausalLM.from_pretrained(streamed_dir, trust_remote_code=True).eval()
    reference = AutoModelForCausalLM.from_pretrained(memory_dir, trust_remote_code=True).eval()
    with torch.no_grad():
        torch.testing.assert_close(
            loaded(input_ids, use_cache=False).logits,
            reference(input_ids, use_cache=False).logits,
        )
