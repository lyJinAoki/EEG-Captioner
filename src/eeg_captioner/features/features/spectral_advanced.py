"""Advanced per-channel spectral features from Welch PSD."""

from __future__ import annotations

from typing import Dict, Mapping, Tuple

import numpy as np
from scipy.integrate import trapezoid
from scipy.signal import welch

from eeg_captioner.features.feature_registry import TrialContext


def _get_welch(
    signal: np.ndarray,
    context: TrialContext,
    window_sec: float,
    overlap: float,
) -> tuple[np.ndarray, np.ndarray]:
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


def _iaf(freqs: np.ndarray, pxx: np.ndarray, lo: float, hi: float) -> np.ndarray:
    mask = (freqs >= lo) & (freqs <= hi)
    sub_f, sub_p = freqs[mask], pxx[:, mask]
    return sub_f[np.argmax(sub_p, axis=-1)]


def _spectral_entropy(
    freqs: np.ndarray, pxx: np.ndarray, lo: float, hi: float, eps: float
) -> np.ndarray:
    mask = (freqs >= lo) & (freqs <= hi)
    p = pxx[:, mask]
    p = p / (p.sum(axis=-1, keepdims=True) + eps)
    h = -(p * np.log(p + eps)).sum(axis=-1)
    return h / np.log(p.shape[-1] + eps)


def _sef(freqs: np.ndarray, pxx: np.ndarray, lo: float, hi: float, frac: float, eps: float) -> np.ndarray:
    mask = (freqs >= lo) & (freqs <= hi)
    f, p = freqs[mask], pxx[:, mask]
    if f.size < 2:
        return np.full(pxx.shape[0], np.nan)
    dx = np.diff(f)
    cum = np.cumsum(0.5 * (p[:, :-1] + p[:, 1:]) * dx, axis=1)
    target = frac * (cum[:, -1:] + eps)
    idx = np.argmax(cum >= target, axis=1)
    return f[np.minimum(idx + 1, f.size - 1)]


def _loglog_slope(freqs: np.ndarray, pxx: np.ndarray, lo: float, hi: float, eps: float) -> np.ndarray:
    """Return aperiodic exponent: log10(P) ~ offset - exponent * log10(f)."""
    mask = (freqs >= lo) & (freqs <= hi)
    lf = np.log10(freqs[mask])
    lp = np.log10(pxx[:, mask] + eps)
    lf_c = lf - lf.mean()
    var = float((lf_c**2).sum()) + eps
    return -(lp @ lf_c) / var


def _alpha_osc_power(
    freqs: np.ndarray,
    pxx: np.ndarray,
    fit_lo: float,
    fit_hi: float,
    alpha_lo: float,
    alpha_hi: float,
    eps: float,
) -> np.ndarray:
    mask = (freqs >= fit_lo) & (freqs <= fit_hi)
    lf = np.log10(freqs[mask])
    lp = np.log10(pxx[:, mask] + eps)
    lf_c = lf - lf.mean()
    var = float((lf_c**2).sum()) + eps
    slope = (lp @ lf_c) / var
    intercept = lp.mean(axis=1) - slope * lf.mean()
    log_bg = intercept[:, None] + slope[:, None] * np.log10(freqs[None, :] + eps)
    residual = np.maximum(pxx - np.power(10.0, log_bg), 0.0)
    alpha_mask = (freqs >= alpha_lo) & (freqs <= alpha_hi)
    return trapezoid(residual[:, alpha_mask], freqs[alpha_mask], axis=-1)


def build_spectral_advanced_plugin(config: Mapping[str, object]):
    """Create spectral_advanced feature callable."""
    analysis_range = config.get("analysis_range", [0.5, 45.0])
    analysis_lo, analysis_hi = float(analysis_range[0]), float(analysis_range[1])
    alpha_range = config.get("alpha_range", [8.0, 13.0])
    alpha_lo, alpha_hi = float(alpha_range[0]), float(alpha_range[1])
    fit_range = config.get("fooof_fit_range", [2.0, 40.0])
    fit_lo, fit_hi = float(fit_range[0]), float(fit_range[1])
    welch_window_sec = float(config.get("welch_window_sec", 4.0))
    welch_overlap = float(config.get("welch_overlap", 0.5))
    sef_percent = float(config.get("sef_percent", 95.0)) / 100.0
    eps = float(config.get("eps", 1e-12))
    theta = tuple(config.get("theta_band", [4.0, 8.0]))  # type: ignore[assignment]
    alpha = tuple(config.get("alpha_band", [8.0, 13.0]))  # type: ignore[assignment]
    beta = tuple(config.get("beta_band", [13.0, 30.0]))  # type: ignore[assignment]

    def compute(signal: np.ndarray, context: TrialContext) -> Dict[str, np.ndarray]:
        freqs, pxx = _get_welch(signal, context, welch_window_sec, welch_overlap)
        p_theta = _band_power(freqs, pxx, float(theta[0]), float(theta[1]))
        p_alpha = _band_power(freqs, pxx, float(alpha[0]), float(alpha[1]))
        p_beta = _band_power(freqs, pxx, float(beta[0]), float(beta[1]))

        outputs: Dict[str, np.ndarray] = {
            "ratio_theta_beta": (p_theta / (p_beta + eps)).astype(np.float32),
            "ratio_alpha_theta": (p_alpha / (p_theta + eps)).astype(np.float32),
            "iaf_hz": _iaf(freqs, pxx, alpha_lo, alpha_hi).astype(np.float32),
            "spectral_entropy": _spectral_entropy(freqs, pxx, analysis_lo, analysis_hi, eps).astype(
                np.float32
            ),
            "sef_hz": _sef(freqs, pxx, analysis_lo, analysis_hi, sef_percent, eps).astype(np.float32),
            "aperiodic_exponent": _loglog_slope(freqs, pxx, fit_lo, fit_hi, eps).astype(np.float32),
            "alpha_oscillatory_power": _alpha_osc_power(
                freqs, pxx, fit_lo, fit_hi, alpha_lo, alpha_hi, eps
            ).astype(np.float32),
        }
        # Per-channel features keep raw values for every channel (bad channels
        # only corrupt their own row, not others). Bad-channel info lives in
        # meta/bad_channel_mask; downstream consumers apply it as needed. Joint
        # features (see features/spatial.py) still mask internally.
        return outputs

    return compute
