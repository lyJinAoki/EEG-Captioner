"""Fixed-length trial segmentation for feature extraction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Mapping, Tuple

import numpy as np


@dataclass(frozen=True)
class SegmentSpec:
    duration_sec: float = 10.0
    overlap_sec: float = 0.0
    min_duration_sec: float = 10.0

    @classmethod
    def from_config(cls, cfg: Mapping[str, object] | None) -> SegmentSpec | None:
        if not cfg or not bool(cfg.get("enabled", False)):
            return None
        return cls(
            duration_sec=float(cfg.get("duration_sec", 10.0)),
            overlap_sec=float(cfg.get("overlap_sec", 0.0)),
            min_duration_sec=float(cfg.get("min_duration_sec", cfg.get("duration_sec", 10.0))),
        )


def iter_segments(
    signal: np.ndarray,
    sample_rate: float,
    spec: SegmentSpec,
) -> Iterator[Tuple[np.ndarray, float, float]]:
    """Yield (segment_signal, start_sec, duration_sec) from one trial."""
    n_samples = int(signal.shape[1])
    win = max(1, int(round(spec.duration_sec * sample_rate)))
    min_len = max(1, int(round(spec.min_duration_sec * sample_rate)))
    step = max(1, int(round((spec.duration_sec - spec.overlap_sec) * sample_rate)))

    start = 0
    while start + min_len <= n_samples:
        end = min(start + win, n_samples)
        seg_len = end - start
        if seg_len < min_len:
            break
        yield signal[:, start:end], start / sample_rate, seg_len / sample_rate
        if end >= n_samples:
            break
        start += step


def iter_equal_parts(
    signal: np.ndarray,
    sample_rate: float,
    n_parts: int,
) -> Iterator[Tuple[np.ndarray, float, float]]:
    """Yield near-equal contiguous parts as (part_signal, offset_sec, duration_sec)."""
    if n_parts <= 0:
        raise ValueError("n_parts must be positive.")
    n_samples = int(signal.shape[1])
    boundaries = np.linspace(0, n_samples, n_parts + 1, dtype=int)
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        part = signal[:, start:end]
        yield part, start / sample_rate, (end - start) / sample_rate
