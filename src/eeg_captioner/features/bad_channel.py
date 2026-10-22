"""Bad-channel preprocessing for per-trial EEG feature extraction."""

from __future__ import annotations

from typing import Any, List, Mapping, Sequence

import numpy as np

from eeg_captioner.features.feature_registry import TrialContext

_MNE_MODULE: Any | None = None
_MNE_ANNOTATE_AMPLITUDE: Any | None = None
_MNE_IMPORT_ERROR: Exception | None = None


def _load_mne() -> tuple[Any | None, Any | None]:
    global _MNE_MODULE, _MNE_ANNOTATE_AMPLITUDE, _MNE_IMPORT_ERROR
    if _MNE_MODULE is not None and _MNE_ANNOTATE_AMPLITUDE is not None:
        return _MNE_MODULE, _MNE_ANNOTATE_AMPLITUDE
    if _MNE_IMPORT_ERROR is not None:
        return None, None
    try:
        import mne
        from mne.preprocessing import annotate_amplitude

        _MNE_MODULE = mne
        _MNE_ANNOTATE_AMPLITUDE = annotate_amplitude
        return _MNE_MODULE, _MNE_ANNOTATE_AMPLITUDE
    except Exception as exc:  # noqa: BLE001
        _MNE_IMPORT_ERROR = exc
        return None, None


def _as_float(config: Mapping[str, object], key: str, default: float) -> float:
    value = config.get(key, default)
    try:
        return float(value)  # type: ignore[arg-type]
    except Exception:
        return default


def _as_str_list(value: object) -> List[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    return [str(x).strip() for x in value if str(x).strip()]


def _detect_bad_channels_with_mne(
    signal: np.ndarray,
    context: TrialContext,
    config: Mapping[str, object],
) -> tuple[np.ndarray, float]:
    channel_names = [str(name) for name in context.channel_names]
    always_bad = {x.upper() for x in _as_str_list(config.get("always_bad_channels", []))}
    static_mask = np.array([name.upper() in always_bad for name in channel_names], dtype=bool)

    mne, annotate_amplitude = _load_mne()
    if mne is None or annotate_amplitude is None:
        raise RuntimeError(
            "Configured bad-channel detection requires MNE; install "
            f"eeg-captioner[preprocess]. Import error: {_MNE_IMPORT_ERROR!s}"
        )

    eval_picks = np.array([idx for idx, bad in enumerate(static_mask) if not bad], dtype=int)
    if eval_picks.size == 0:
        return static_mask, 0.0

    # Input signal is in microvolts; MNE Raw uses volts.
    info = mne.create_info(ch_names=channel_names, sfreq=context.sample_rate, ch_types="eeg")
    raw = mne.io.RawArray((signal * 1e-6).astype(np.float64, copy=False), info, verbose="ERROR")

    peak_uv = _as_float(config, "peak_threshold_uv", 500.0)
    flat_uv = _as_float(config, "flat_threshold_uv", 1.0)
    bad_percent = _as_float(config, "bad_percent_threshold", 5.0)
    min_bad_duration_sec = _as_float(config, "min_bad_duration_sec", 0.02)
    peak_v = peak_uv * 1e-6 if peak_uv > 0 else None
    flat_v = flat_uv * 1e-6 if flat_uv > 0 else None

    try:
        annotations, bad_channels = annotate_amplitude(
            raw,
            peak=peak_v,
            flat=flat_v,
            bad_percent=bad_percent,
            min_duration=min_bad_duration_sec,
            picks=eval_picks.tolist(),
            verbose="ERROR",
        )
    except TypeError:
        annotations, bad_channels = annotate_amplitude(
            raw,
            peak=peak_v,
            flat=flat_v,
            bad_percent=bad_percent,
            picks=eval_picks.tolist(),
            verbose="ERROR",
        )

    bad_set = {str(name).upper() for name in bad_channels}
    detected_mask = np.array([name.upper() in bad_set for name in channel_names], dtype=bool)
    combined_mask = static_mask | detected_mask

    trial_duration = signal.shape[1] / max(context.sample_rate, 1e-6)
    bad_duration = 0.0
    for desc, duration in zip(annotations.description, annotations.duration):
        if str(desc).upper().startswith("BAD_"):
            bad_duration += float(duration)
    bad_time_ratio = bad_duration / max(trial_duration, 1e-6)
    return combined_mask, bad_time_ratio


def segment_exceeds_hard_peak(
    seg_signal: np.ndarray,
    bad_channel_mask: np.ndarray | None,
    *,
    threshold_uv: float = 100.0,
) -> bool:
    """Return True if any good-channel sample exceeds |V| > threshold_uv.

    Bad channels in ``bad_channel_mask`` are excluded from the check. An empty
    good-channel set is treated as exceeding (segment should be rejected).
    """
    if seg_signal.size == 0:
        return True
    if bad_channel_mask is None:
        good = np.asarray(seg_signal, dtype=np.float64)
    else:
        mask = np.asarray(bad_channel_mask, dtype=bool)
        if mask.shape[0] != seg_signal.shape[0]:
            raise ValueError(
                f"bad_channel_mask length {mask.shape[0]} != n_channels {seg_signal.shape[0]}"
            )
        if mask.all():
            return True
        good = np.asarray(seg_signal[~mask], dtype=np.float64)
    if good.size == 0:
        return True
    return bool(np.any(np.abs(good) > float(threshold_uv)))


def bad_channel_preprocess(
    signal: np.ndarray,
    context: TrialContext,
    *,
    enabled: bool = False,
    config: Mapping[str, object] | None = None,
) -> np.ndarray:
    """Mark bad channels/trials with MNE and optionally zero bad channels."""
    if not enabled:
        context.bad_channel_mask = np.zeros(signal.shape[0], dtype=bool)
        context.bad_trial = False
        context.bad_trial_reason = ""
        return signal

    cfg = config or {}
    mask, bad_time_ratio = _detect_bad_channels_with_mne(signal, context, cfg)
    context.bad_channel_mask = mask

    bad_channel_ratio = float(mask.mean()) if mask.size else 0.0
    max_bad_channel_ratio = _as_float(cfg, "max_bad_channel_ratio_for_trial", 0.3)
    max_bad_time_ratio = _as_float(cfg, "max_bad_time_ratio_for_trial", 0.2)

    trial_reasons: List[str] = []
    if bad_channel_ratio > max_bad_channel_ratio:
        trial_reasons.append(
            f"bad_channel_ratio={bad_channel_ratio:.3f}>{max_bad_channel_ratio:.3f}"
        )
    if bad_time_ratio > max_bad_time_ratio:
        trial_reasons.append(f"bad_time_ratio={bad_time_ratio:.3f}>{max_bad_time_ratio:.3f}")
    if trial_reasons:
        context.bad_trial = True
        context.bad_trial_reason = "; ".join(trial_reasons)
    else:
        context.bad_trial = False
        context.bad_trial_reason = ""

    bad_names = [name for name, is_bad in zip(context.channel_names, mask) if is_bad]
    if bad_names:
        preview = ", ".join(bad_names[:8])
        suffix = " ..." if len(bad_names) > 8 else ""
        context.warnings.append(
            f"trial={context.trial_id} bad_channels={len(bad_names)} [{preview}{suffix}]"
        )
    if context.bad_trial:
        context.warnings.append(f"trial={context.trial_id} marked_bad_trial: {context.bad_trial_reason}")

    action = str(cfg.get("action", "mark")).strip().lower()
    if action == "zero" and mask.any():
        patched = np.asarray(signal, dtype=np.float64).copy()
        patched[mask, :] = 0.0
        return patched
    return signal
