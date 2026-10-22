"""Yeo-7 source-network bandpower features via MNE source localization."""

from __future__ import annotations

from typing import Dict, Mapping, Tuple

import numpy as np
from scipy.integrate import trapezoid
from scipy.signal import periodogram

from eeg_captioner.features.feature_registry import TrialContext
from eeg_captioner.features.features.bandpower import DEFAULT_BANDS, _normalize_bands
from eeg_captioner.features.source_artifacts import (
    YEO7_NETWORK_NAMES,
    get_source_artifacts,
    mne_import_error,
    select_source_channel_indices,
    to_mne_channel_name,
)


def _nan_network_outputs(n_bands: int) -> Dict[str, np.ndarray]:
    n_net = len(YEO7_NETWORK_NAMES)
    nan_band = np.full((n_net, n_bands), np.nan, dtype=np.float32)
    return {
        "abs_power": nan_band.copy(),
        "rel_power": nan_band.copy(),
        "total_power_0p5_45": np.full(n_net, np.nan, dtype=np.float32),
        "n_epochs_used": np.float32(np.nan),
    }


def _band_power(freqs: np.ndarray, pxx: np.ndarray, lo: float, hi: float) -> np.ndarray:
    mask = (freqs >= lo) & (freqs < hi)
    return trapezoid(pxx[:, mask], freqs[mask], axis=-1)


def build_source_network_plugin(config: Mapping[str, object]):
    """Create source-network feature callable."""
    bands = _normalize_bands(config.get("bands"))  # type: ignore[arg-type]
    band_names = list(bands.keys())
    n_bands = len(band_names)
    n_networks = len(YEO7_NETWORK_NAMES)

    analysis_range = config.get("analysis_range", [0.5, 45.0])
    analysis_lo = float(analysis_range[0])
    analysis_hi = float(analysis_range[1])
    epoch_duration_sec = float(config.get("epoch_duration_sec", 4.0))
    epoch_overlap = float(config.get("epoch_overlap", 0.5))
    max_epochs = int(config.get("max_epochs", 30))
    min_channels = int(config.get("min_channels", len(config.get("channels", ()))))
    eps = float(config.get("eps", 1e-30))

    artifacts_cfg = {
        "artifacts_dir": config.get("artifacts_dir", "artifacts/source"),
        "sample_rate_hz": config.get("sample_rate_hz", 200.0),
        "channels": config.get("channels"),
        "inverse_method": config.get("inverse_method", "eLORETA"),
        "noise_cov_method": config.get("noise_cov_method", "ad_hoc"),
        "montage": config.get("montage", "standard_1020"),
        "source_spacing": config.get("source_spacing", "ico4"),
        "head_model": config.get("head_model", "bem"),
        "loose": config.get("loose", 0.2),
        "depth": config.get("depth", 0.8),
    }

    def compute(signal: np.ndarray, context: TrialContext) -> Dict[str, np.ndarray]:
        cache_key = f"source_network:net_ts:{signal.shape[1]}"
        cached = context.cache.get(cache_key)
        if cached is None:
            if mne_import_error() is not None:
                context.warnings.append(
                    f"source_network skipped: mne unavailable ({mne_import_error()!s})"
                )
                return _nan_network_outputs(n_bands)

            try:
                artifacts = get_source_artifacts(artifacts_cfg)
            except Exception as exc:  # noqa: BLE001
                context.warnings.append(f"source_network artifacts unavailable: {exc}")
                return _nan_network_outputs(n_bands)

            import mne

            indices, mne_names = select_source_channel_indices(
                context.channel_names,
                bad_channel_mask=context.bad_channel_mask,
            )
            if len(indices) < min_channels:
                context.warnings.append(
                    f"source_network skipped: insufficient good channels "
                    f"(need {min_channels})"
                )
                return _nan_network_outputs(n_bands)

            required = [str(c) for c in artifacts.channel_names]
            picked_map = {name.upper(): i for i, name in enumerate(mne_names)}
            ordered_rows = []
            for ch in required:
                row = picked_map.get(ch.upper()) or picked_map.get(to_mne_channel_name(ch).upper())
                if row is None:
                    context.warnings.append(
                        f"source_network unit={context.unit_id}: missing required channel {ch}"
                    )
                    return _nan_network_outputs(n_bands)
                ordered_rows.append(indices[row])

            aligned = signal[np.array(ordered_rows, dtype=int), :].astype(np.float64, copy=False)
            info = mne.create_info(required, sfreq=context.sample_rate, ch_types="eeg")
            raw = mne.io.RawArray((aligned * 1e-6), info, verbose="ERROR")
            raw.set_montage(str(artifacts_cfg.get("montage", "standard_1020")), verbose="ERROR")
            # MNE inverse modeling requires an average-reference projector.
            raw.set_eeg_reference("average", projection=True, verbose="ERROR")

            try:
                raw_duration_sec = aligned.shape[1] / max(context.sample_rate, 1e-6)
                if raw_duration_sec < epoch_duration_sec:
                    epochs = mne.EpochsArray(
                        raw.get_data()[None, :, :],
                        raw.info.copy(),
                        tmin=0.0,
                        verbose="ERROR",
                    )
                else:
                    epochs = mne.make_fixed_length_epochs(
                        raw,
                        duration=epoch_duration_sec,
                        overlap=epoch_overlap,
                        preload=True,
                        verbose="ERROR",
                    )
            except Exception as exc:  # noqa: BLE001
                context.warnings.append(f"source_network epoching failed: {exc}")
                return _nan_network_outputs(n_bands)

            if len(epochs) == 0:
                context.warnings.append(f"source_network unit={context.unit_id}: no epochs")
                return _nan_network_outputs(n_bands)

            epochs.apply_proj(verbose="ERROR")

            if len(epochs) > max_epochs:
                pick = np.linspace(0, len(epochs) - 1, max_epochs, dtype=int)
                epochs = epochs[pick]

            try:
                stcs = mne.minimum_norm.apply_inverse_epochs(
                    epochs,
                    artifacts.inv,
                    lambda2=1.0 / 9.0,
                    method=str(artifacts.inverse_method),
                    pick_ori="normal",
                    verbose="ERROR",
                )
                label_ts = mne.extract_label_time_course(
                    stcs,
                    artifacts.labels,
                    artifacts.src,
                    mode="mean_flip",
                    return_generator=False,
                    verbose="ERROR",
                )
            except Exception as exc:  # noqa: BLE001
                context.warnings.append(f"source_network inverse failed: {exc}")
                return _nan_network_outputs(n_bands)

            # label_ts: (n_labels, n_epochs, n_times)
            net_ts = np.asarray(label_ts, dtype=np.float64).reshape(n_networks, -1)
            cached = (net_ts, np.float32(len(epochs)))
            context.cache[cache_key] = cached

        net_ts, n_epochs_used = cached  # type: ignore[misc]
        net_ts = np.asarray(net_ts, dtype=np.float64)
        if context.segment_part_name:
            boundaries = np.linspace(0, net_ts.shape[1], 4, dtype=int)
            part_index = int(context.segment_part_index)
            start = int(boundaries[part_index])
            end = int(boundaries[part_index + 1])
            net_ts = net_ts[:, start:end]
        if net_ts.shape[1] < 16:
            context.warnings.append(
                f"source_network unit={context.unit_id}: network time series too short"
            )
            return _nan_network_outputs(n_bands)

        freqs, pxx = periodogram(
            net_ts,
            fs=context.sample_rate,
            axis=-1,
        )

        analysis_mask = (freqs >= analysis_lo) & (freqs <= analysis_hi)
        total_power = trapezoid(pxx[:, analysis_mask], freqs[analysis_mask], axis=-1) + eps

        band_arrays = []
        for band_name in band_names:
            lo, hi = bands[band_name]
            band_arrays.append(_band_power(freqs, pxx, lo, hi))
        abs_power = np.stack(band_arrays, axis=-1).astype(np.float32)

        return {
            "abs_power": abs_power,
            "rel_power": (abs_power / total_power[:, None]).astype(np.float32),
            "total_power_0p5_45": total_power.astype(np.float32),
            "n_epochs_used": np.float32(n_epochs_used),
        }

    return compute
