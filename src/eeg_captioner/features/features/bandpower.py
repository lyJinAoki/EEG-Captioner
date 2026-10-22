"""Bandpower feature plugin using Welch PSD."""

from __future__ import annotations

from typing import Dict, Iterable, Mapping, Tuple

import numpy as np
from scipy.integrate import trapezoid
from scipy.signal import welch

from eeg_captioner.features.feature_registry import TrialContext


DEFAULT_BANDS: Dict[str, Tuple[float, float]] = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
}


def _normalize_bands(
    raw_bands: Mapping[str, Iterable[float]] | None,
) -> Dict[str, Tuple[float, float]]:
    if not raw_bands:
        return dict(DEFAULT_BANDS)
    bands: Dict[str, Tuple[float, float]] = {}
    for name, bounds in raw_bands.items():
        lo, hi = list(bounds)
        bands[name] = (float(lo), float(hi))
    return bands


def build_bandpower_plugin(config: Mapping[str, object]):
    """Create a configured bandpower feature callable."""
    bands = _normalize_bands(config.get("bands"))  # type: ignore[arg-type]
    analysis_range = config.get("analysis_range", [0.5, 45.0])
    analysis_lo = float(analysis_range[0])
    analysis_hi = float(analysis_range[1])
    welch_window_sec = float(config.get("welch_window_sec", 4.0))
    welch_overlap = float(config.get("welch_overlap", 0.5))
    return_relative = bool(config.get("relative", True))
    return_log_abs = bool(config.get("log_power", False))
    eps = float(config.get("eps", 1e-12))

    def compute(signal: np.ndarray, context: TrialContext) -> Dict[str, np.ndarray]:
        n_samples = signal.shape[1]
        nperseg = max(16, min(int(context.sample_rate * welch_window_sec), n_samples))
        noverlap = int(nperseg * welch_overlap)
        if noverlap >= nperseg:
            noverlap = nperseg // 2

        cache_key = f"welch:{nperseg}:{noverlap}:part{int(context.segment_part_index)}"
        cached = context.cache.get(cache_key)
        if cached is None:
            freqs, pxx = welch(
                signal,
                fs=context.sample_rate,
                nperseg=nperseg,
                noverlap=noverlap,
                axis=-1,
            )
            context.cache[cache_key] = (freqs, pxx)
        else:
            freqs, pxx = cached  # type: ignore[misc]

        # Welch 输出频率点等间隔，频带功率用 PSD 求和 × 频率间隔计算
        df = float(freqs[1] - freqs[0])

        analysis_mask = (freqs >= analysis_lo) & (freqs <= analysis_hi)
        total_power = (
            np.sum(pxx[:, analysis_mask], axis=-1) * df
            + eps
        )

        ordered_band_names = list(bands.keys())
        band_arrays = []

        for band_name in ordered_band_names:
            lo, hi = bands[band_name]

            # 使用左闭右开区间，避免相邻频带重复计算边界频点
            mask = (freqs >= lo) & (freqs < hi)

            band_power = np.sum(
                pxx[:, mask],
                axis=-1,
            ) * df

            band_arrays.append(band_power)

        abs_power = np.stack(band_arrays, axis=-1).astype(np.float32)
        
        outputs: Dict[str, np.ndarray] = {
            "abs_power": abs_power,
            "total_power_0p5_45": total_power.astype(np.float32),
        }
        if return_relative:
            outputs["rel_power"] = (abs_power / total_power[:, None]).astype(np.float32)
        if return_log_abs:
            outputs["log_abs_power"] = np.log10(abs_power + eps).astype(np.float32)
        return outputs

    return compute
