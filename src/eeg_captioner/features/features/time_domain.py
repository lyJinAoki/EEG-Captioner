"""Time-domain feature plugin (per-channel statistics on raw trial signal)."""

from __future__ import annotations

from typing import Dict, Mapping

import numpy as np
from scipy.stats import kurtosis, skew

from eeg_captioner.features.feature_registry import TrialContext

DEFAULT_METRICS = (
    "mean_amplitude",
    "variance",
    "std",
    "rms",
    "peak_to_peak",
    "skewness",
    "kurtosis",
    "zero_crossing_rate",
    "line_length",
    "hjorth_activity",
    "hjorth_mobility",
    "hjorth_complexity",
)


def _hjorth_mobility(signal: np.ndarray, eps: float) -> np.ndarray:
    diff = np.diff(signal, axis=1)
    var_signal = np.var(signal, axis=1)
    var_diff = np.var(diff, axis=1)
    return np.sqrt(var_diff / (var_signal + eps))


def build_time_domain_plugin(config: Mapping[str, object]):
    """Create a configured time-domain feature callable."""
    eps = float(config.get("eps", 1e-12))
    requested = config.get("metrics")
    if requested:
        metrics = tuple(str(name) for name in requested)  # type: ignore[union-attr]
    else:
        metrics = DEFAULT_METRICS
    unknown = set(metrics) - set(DEFAULT_METRICS)
    if unknown:
        raise ValueError(f"Unknown time_domain metrics: {sorted(unknown)}")

    def compute(signal: np.ndarray, context: TrialContext) -> Dict[str, np.ndarray]:
        del context  # per-trial stats need only the signal array
        n_samples = signal.shape[1]
        outputs: Dict[str, np.ndarray] = {}

        if "mean_amplitude" in metrics:
            outputs["mean_amplitude"] = np.mean(signal, axis=1).astype(np.float32)
        if "variance" in metrics:
            outputs["variance"] = np.var(signal, axis=1).astype(np.float32)
        if "std" in metrics:
            outputs["std"] = np.std(signal, axis=1).astype(np.float32)
        if "rms" in metrics:
            outputs["rms"] = np.sqrt(np.mean(signal * signal, axis=1) + eps).astype(np.float32)
        if "peak_to_peak" in metrics:
            outputs["peak_to_peak"] = np.ptp(signal, axis=1).astype(np.float32)
        if "skewness" in metrics:
            outputs["skewness"] = skew(signal, axis=1, nan_policy="omit").astype(np.float32)
        if "kurtosis" in metrics:
            outputs["kurtosis"] = kurtosis(signal, axis=1, nan_policy="omit").astype(np.float32)
        if "zero_crossing_rate" in metrics:
            if n_samples > 1:
                sign_changes = np.abs(np.diff(np.sign(signal), axis=1)) > 0
                zcr = sign_changes.sum(axis=1) / float(n_samples - 1)
            else:
                zcr = np.zeros(signal.shape[0], dtype=np.float64)
            outputs["zero_crossing_rate"] = zcr.astype(np.float32)
        if "line_length" in metrics:
            if n_samples > 1:
                ll = np.sum(np.abs(np.diff(signal, axis=1)), axis=1)
            else:
                ll = np.zeros(signal.shape[0], dtype=np.float64)
            outputs["line_length"] = ll.astype(np.float32)
        if "hjorth_activity" in metrics:
            outputs["hjorth_activity"] = np.var(signal, axis=1).astype(np.float32)
        if "hjorth_mobility" in metrics:
            outputs["hjorth_mobility"] = _hjorth_mobility(signal, eps).astype(np.float32)
        if "hjorth_complexity" in metrics:
            mobility = _hjorth_mobility(signal, eps)
            diff = np.diff(signal, axis=1)
            mobility_diff = _hjorth_mobility(diff, eps) if diff.shape[1] > 0 else np.zeros(signal.shape[0])
            outputs["hjorth_complexity"] = (mobility_diff / (mobility + eps)).astype(np.float32)

        return outputs

    return compute
