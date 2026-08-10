"""Convert calibration sums to Winnow and REAP scores."""

from __future__ import annotations

import torch


def reap_scores(stats: dict) -> torch.Tensor:
    """Return normalized-top-k REAP scores with shape ``[L, E]``."""
    counts = stats["token_count"].to(torch.float32)
    sums = stats["reap_sum"].to(torch.float32)
    denominator = counts.clamp_min(1)
    return torch.where(counts > 0, sums / denominator, torch.zeros_like(sums))


def channel_scores(stats: dict) -> torch.Tensor:
    """Return actual-router-weight channel scores with shape ``[L, E, C]``."""
    counts = stats["token_count"].to(torch.float32)
    sums = stats["channel_sum"].to(torch.float32)
    norms = stats["down_norm"].to(torch.float32)
    denominator = counts.clamp_min(1).unsqueeze(-1)
    scores = sums * norms / denominator
    return torch.where(counts.unsqueeze(-1) > 0, scores, torch.zeros_like(scores))
