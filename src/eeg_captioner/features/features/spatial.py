"""Spatial / asymmetry features (§1.5) derived from per-channel band power.

All features here are spatial *derivations* on top of per-channel Welch band
power; they do not introduce new signal-level computations. Outputs:

- ``roi_abs_power`` / ``roi_rel_power``: 5-region scalp ROI averages, shape
  ``(n_roi, n_bands)``.
- ``asymmetry_log``: left-right log-power difference ``log(R) - log(L)`` for
  every symmetric electrode pair, shape ``(n_pairs, n_bands)`` (the frontal
  alpha entry is the classic FAA).
- ``lateralization_index``: normalized ``(R - L) / (R + L)`` for every pair,
  shape ``(n_pairs, n_bands)``.
- one scalar dataset per named anterior-posterior ratio (default
  ``ap_theta_alpha`` and ``ap_alpha``).

Channels are matched by name (case-insensitive) against ``context.channel_names``
so the plugin tolerates files with fewer than the full 24 LaBraM channels.
Missing or bad channels propagate as NaN (handled by the pipeline sanity check).
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np
from scipy.integrate import trapezoid
from scipy.signal import welch

from eeg_captioner.features.feature_registry import TrialContext

# Canonical 5-band definitions (kept consistent with bandpower defaults).
DEFAULT_BANDS: Dict[str, Tuple[float, float]] = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
}

# 5-region scalp ROI grouping (lobe-based), channel names per data_explain.md.
# Reference-like channels (A1/A2/T1/T2) are intentionally excluded.
DEFAULT_ROIS: Dict[str, List[str]] = {
    "frontal": ["FP1", "FP2", "F3", "F4", "F7", "F8", "FZ"],
    "central": ["C3", "C4", "CZ"],
    "parietal": ["P3", "P4", "PZ"],
    "occipital": ["O1", "O2", "OZ"],
    "temporal": ["T3", "T4", "T5", "T6"],
}

# Symmetric left-right electrode pairs as (left, right). Midline electrodes
# (FZ/CZ/PZ/OZ) have no contralateral partner and are excluded.
DEFAULT_PAIRS: List[Tuple[str, str]] = [
    ("FP1", "FP2"),
    ("F3", "F4"),
    ("F7", "F8"),
    ("C3", "C4"),
    ("P3", "P4"),
    ("O1", "O2"),
    ("T3", "T4"),
    ("T5", "T6"),
]

# Named anterior-posterior ratios:
# name -> (numerator_roi, numerator_band, denominator_roi, denominator_band).
DEFAULT_AP_RATIOS: Dict[str, Tuple[str, str, str, str]] = {
    "ap_theta_alpha": ("frontal", "theta", "parietal", "alpha"),
    "ap_alpha": ("frontal", "alpha", "occipital", "alpha"),
}


def pair_labels(pairs: Sequence[Tuple[str, str]] = DEFAULT_PAIRS) -> List[str]:
    """Human-readable labels for symmetric pairs, e.g. ``F3-F4``."""
    return [f"{left}-{right}" for left, right in pairs]


def _normalize_bands(
    raw_bands: Mapping[str, Sequence[float]] | None,
) -> Dict[str, Tuple[float, float]]:
    if not raw_bands:
        return dict(DEFAULT_BANDS)
    bands: Dict[str, Tuple[float, float]] = {}
    for name, bounds in raw_bands.items():
        lo, hi = list(bounds)
        bands[name] = (float(lo), float(hi))
    return bands


def _get_welch(
    signal: np.ndarray,
    context: TrialContext,
    window_sec: float,
    overlap: float,
) -> Tuple[np.ndarray, np.ndarray]:
    n_samples = signal.shape[1]
    nperseg = max(16, min(int(context.sample_rate * window_sec), n_samples))
    noverlap = int(nperseg * overlap)
    if noverlap >= nperseg:
        noverlap = nperseg // 2
    key = f"welch:{nperseg}:{noverlap}:part{int(context.segment_part_index)}"
    cached = context.cache.get(key)
    if cached is None:
        freqs, pxx = welch(
            signal,
            fs=context.sample_rate,
            nperseg=nperseg,
            noverlap=noverlap,
            axis=-1,
        )
        context.cache[key] = (freqs, pxx)
        return freqs, pxx
    return cached  # type: ignore[return-value]


def _band_power(freqs: np.ndarray, pxx: np.ndarray, lo: float, hi: float) -> np.ndarray:
    mask = (freqs >= lo) & (freqs < hi)
    return trapezoid(pxx[:, mask], freqs[mask], axis=-1)


def build_spatial_plugin(config: Mapping[str, object]):
    """Create a configured spatial-feature callable."""
    bands = _normalize_bands(config.get("bands"))  # type: ignore[arg-type]
    band_names = list(bands.keys())

    analysis_range = config.get("analysis_range", [0.5, 45.0])
    analysis_lo, analysis_hi = float(analysis_range[0]), float(analysis_range[1])  # type: ignore[index]
    welch_window_sec = float(config.get("welch_window_sec", 4.0))
    welch_overlap = float(config.get("welch_overlap", 0.5))
    eps = float(config.get("eps", 1e-12))
    # A ROI average is only emitted when at least this many of its channels are
    # present and good; otherwise it is set to NaN ("insufficient evidence").
    roi_min_good_channels = int(config.get("roi_min_good_channels", 1))

    rois_cfg = config.get("rois")
    rois: Dict[str, List[str]] = (
        {str(k): [str(c) for c in v] for k, v in dict(rois_cfg).items()}  # type: ignore[arg-type]
        if rois_cfg
        else {k: list(v) for k, v in DEFAULT_ROIS.items()}
    )
    roi_names = list(rois.keys())

    pairs_cfg = config.get("pairs")
    pairs: List[Tuple[str, str]] = (
        [(str(p[0]), str(p[1])) for p in pairs_cfg]  # type: ignore[index]
        if pairs_cfg
        else list(DEFAULT_PAIRS)
    )

    ap_cfg = config.get("ap_ratios")
    ap_ratios: Dict[str, Tuple[str, str, str, str]] = (
        {str(k): tuple(str(x) for x in v) for k, v in dict(ap_cfg).items()}  # type: ignore[misc]
        if ap_cfg
        else dict(DEFAULT_AP_RATIOS)
    )

    def compute(signal: np.ndarray, context: TrialContext) -> Dict[str, np.ndarray]:
        freqs, pxx = _get_welch(signal, context, welch_window_sec, welch_overlap)
        n_ch = pxx.shape[0]
        n_bands = len(band_names)

        band_power = np.empty((n_ch, n_bands), dtype=np.float64)
        for j, name in enumerate(band_names):
            lo, hi = bands[name]
            band_power[:, j] = _band_power(freqs, pxx, lo, hi)

        analysis_mask = (freqs >= analysis_lo) & (freqs <= analysis_hi)
        total = trapezoid(pxx[:, analysis_mask], freqs[analysis_mask], axis=-1) + eps
        rel_power = band_power / total[:, None]

        # Mask bad channels with NaN so they are excluded from ROI means and
        # invalidate any pair that depends on them.
        mask = context.bad_channel_mask
        if mask is not None and np.asarray(mask, dtype=bool).any():
            bad = np.asarray(mask, dtype=bool)
            band_power[bad, :] = np.nan
            rel_power[bad, :] = np.nan

        name_to_idx = {
            str(ch).strip().upper(): i for i, ch in enumerate(context.channel_names)
        }

        # --- ROI averages -------------------------------------------------
        roi_abs = np.full((len(roi_names), n_bands), np.nan, dtype=np.float64)
        roi_rel = np.full((len(roi_names), n_bands), np.nan, dtype=np.float64)
        roi_n_good = np.zeros(len(roi_names), dtype=np.float64)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            for r, roi in enumerate(roi_names):
                idxs = [name_to_idx[c.strip().upper()] for c in rois[roi] if c.strip().upper() in name_to_idx]
                if not idxs:
                    continue
                # A channel is "good" when present and not NaN (NaN == bad/missing).
                good = ~np.isnan(band_power[idxs, 0])
                n_good = int(good.sum())
                roi_n_good[r] = n_good
                if n_good >= roi_min_good_channels:
                    roi_abs[r] = np.nanmean(band_power[idxs], axis=0)
                    roi_rel[r] = np.nanmean(rel_power[idxs], axis=0)

        # --- Left-right asymmetry and lateralization index ----------------
        asym = np.full((len(pairs), n_bands), np.nan, dtype=np.float64)
        lat_index = np.full((len(pairs), n_bands), np.nan, dtype=np.float64)
        with np.errstate(invalid="ignore", divide="ignore"):
            for p, (left, right) in enumerate(pairs):
                li = name_to_idx.get(left.strip().upper())
                ri = name_to_idx.get(right.strip().upper())
                if li is None or ri is None:
                    continue
                p_left = band_power[li]
                p_right = band_power[ri]
                asym[p] = np.log(p_right + eps) - np.log(p_left + eps)
                lat_index[p] = (p_right - p_left) / (p_right + p_left + eps)

        outputs: Dict[str, np.ndarray] = {
            "roi_abs_power": roi_abs.astype(np.float32),
            "roi_rel_power": roi_rel.astype(np.float32),
            "roi_n_good": roi_n_good.astype(np.float32),
            "asymmetry_log": asym.astype(np.float32),
            "lateralization_index": lat_index.astype(np.float32),
        }

        # --- Anterior-posterior ratios ------------------------------------
        for ap_name, (num_roi, num_band, den_roi, den_band) in ap_ratios.items():
            num = roi_abs[roi_names.index(num_roi), band_names.index(num_band)]
            den = roi_abs[roi_names.index(den_roi), band_names.index(den_band)]
            outputs[ap_name] = np.float32(num / (den + eps))

        return outputs

    return compute
