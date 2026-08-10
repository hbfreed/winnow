"""Retention value validation."""

from __future__ import annotations

import math


def validate_keep(value: float | str) -> float:
    """Return a finite decimal retention fraction in the interval ``(0, 1]``."""
    try:
        keep = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("keep must be a decimal from 0 to 1") from exc
    if not math.isfinite(keep) or not 0.0 < keep <= 1.0:
        raise ValueError("keep must be a finite decimal greater than 0 and at most 1")
    return keep
