#!/usr/bin/env python3
"""Fill EEG summary markdown templates from hierarchical segment H5 outputs."""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import tempfile
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Sequence, Tuple

import h5py
import numpy as np


BANDS = ("delta", "theta", "alpha", "beta", "gamma")
TEMPLATE_CHANNELS = (
    "FP1",
    "FP2",
    "F3",
    "F4",
    "F7",
    "F8",
    "FZ",
    "C3",
    "C4",
    "CZ",
    "P3",
    "P4",
    "PZ",
    "O1",
    "O2",
    "OZ",
    "T3",
    "T4",
    "T5",
    "T6",
)
CHANNEL_ALIASES = {"T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8"}
PERMANENT_BAD_CHANNELS = frozenset({"A1", "A2", "T1", "T2"})
CONTEXT_CHANNELS = TEMPLATE_CHANNELS + ("A1", "A2", "T1", "T2")
ROIS = ("frontal", "central", "parietal", "occipital", "temporal")
ROI_SHORT = {"F": "frontal", "C": "central", "P": "parietal", "O": "occipital", "T": "temporal"}
# Template order (F/C, F/P, ...); do not filter by letter sort — that yields C_F/O_P/O_T keys.
ROI_RATIO_PAIRS = (
    ("F", "C"),
    ("F", "P"),
    ("F", "T"),
    ("F", "O"),
    ("C", "P"),
    ("C", "T"),
    ("C", "O"),
    ("P", "T"),
    ("P", "O"),
    ("T", "O"),
)
PAIRS = (
    "FP1-FP2",
    "F3-F4",
    "F7-F8",
    "C3-C4",
    "P3-P4",
    "O1-O2",
    "T3-T4",
    "T5-T6",
)
PAIR_ALIASES = {"T3-T4": "T7-T8", "T5-T6": "P7-P8"}
SOURCE_NETWORKS = (
    ("Visual", "vis"),
    ("Somatomotor", "sommot"),
    ("Dorsal Attention", "dorsattn"),
    ("Salience/Ventral Attention", "salventattn"),
    ("Limbic", "limbic"),
    ("Control", "cont"),
    ("Default Mode", "default"),
)
SPECTRAL_CHANGE_KEYS = {
    "dtb": "ratio_theta_beta",
    "dat": "ratio_alpha_theta",
    "diaf": "iaf_hz",
    "dentropy": "spectral_entropy",
    "dsef": "sef_hz",
    "dexp": "aperiodic_exponent",
    "dalphaosc": "alpha_oscillatory_power",
}
TIME_CHANGE_KEYS = {
    "dmean": "mean_amplitude",
    "dstd": "std",
    "drms": "rms",
    "dp2p": "peak_to_peak",
    "dzcr": "zero_crossing_rate",
    "dll": "line_length",
    "dmob": "hjorth_mobility",
    "dcomp": "hjorth_complexity",
}


@dataclass(frozen=True)
class SegmentRef:
    h5_path: Path
    trial_index: int
    segment_index: int


class SafeDict(dict):
    def __missing__(self, key: str) -> str:
        return "NaN"


class H5ReadCache:
    """Cache HDF5 handles and the current segment's small feature arrays."""

    def __init__(self, h5: h5py.File):
        self.h5 = h5
        self.attrs = h5.attrs
        self._datasets: Dict[str, Any] = {}
        self._segment_key: Tuple[int, int] | None = None
        self._segment_arrays: Dict[str, np.ndarray] = {}
        self._trial_arrays: Dict[Tuple[str, int], np.ndarray] = {}

    def __contains__(self, path: str) -> bool:
        return path in self._datasets or path in self.h5

    def __getitem__(self, path: str) -> Any:
        if path not in self._datasets:
            self._datasets[path] = self.h5[path]
        return self._datasets[path]

    def begin_segment(self, ref: SegmentRef) -> None:
        key = (ref.trial_index, ref.segment_index)
        if key != self._segment_key:
            self._segment_key = key
            self._segment_arrays.clear()

    def segment_array(self, path: str, ref: SegmentRef) -> np.ndarray | None:
        self.begin_segment(ref)
        if path not in self:
            return None
        if path not in self._segment_arrays:
            self._segment_arrays[path] = np.asarray(
                self[path][ref.trial_index, ref.segment_index]
            )
        return self._segment_arrays[path]

    def trial_array(self, path: str, trial_index: int) -> np.ndarray | None:
        if path not in self:
            return None
        key = (path, trial_index)
        if key not in self._trial_arrays:
            self._trial_arrays[key] = np.asarray(self[path][trial_index])
        return self._trial_arrays[key]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fill summary markdown templates from H5 outputs.")
    parser.add_argument(
        "--input-glob",
        default="outputs/features_segment_10s_thirds_v2/*.h5",
        help="Glob for hierarchical thirds H5 files.",
    )
    parser.add_argument("--template-v1", default=str(Path(__file__).parent / "templates/summary_template_v1.md"))
    parser.add_argument("--template-v2", default=str(Path(__file__).parent / "templates/summary_template_v2.md"))
    parser.add_argument(
        "--output-dir",
        default="outputs/summary",
        help="Root directory; each H5 batch writes under {start}-{end}/.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Number of H5 files per summary batch folder (default: 1000).",
    )
    parser.add_argument(
        "--batch-index",
        type=int,
        default=None,
        help="Only process this zero-based batch index. Default: process all batches.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Cap the sorted H5 list to the first N files (smoke tests).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max good segments to render across selected files. Default: no limit.",
    )
    parser.add_argument(
        "--trial-index",
        type=int,
        default=None,
        help="Only render segments from this zero-based trial index.",
    )
    parser.add_argument(
        "--segment-index",
        type=int,
        default=None,
        help="Only render this zero-based segment index.",
    )
    parser.add_argument(
        "--keep-existing",
        action="store_true",
        help="Do not remove previously generated markdown files in the batch output dir first.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Preserve complete existing segment outputs, repair partial segments, "
            "and render only missing segments."
        ),
    )
    return parser.parse_args()


def decode_array(values: np.ndarray) -> List[str]:
    out: List[str] = []
    for value in values:
        if isinstance(value, bytes):
            out.append(value.decode("utf-8", errors="replace"))
        else:
            out.append(str(value))
    return out


def load_json_attr(attrs: h5py.AttributeManager, key: str, fallback: Sequence[str]) -> List[str]:
    raw = attrs.get(key)
    if raw is None:
        return list(fallback)
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", errors="replace")
    try:
        value = json.loads(str(raw))
    except json.JSONDecodeError:
        return list(fallback)
    if isinstance(value, list):
        return [str(x) for x in value]
    if isinstance(value, dict):
        return [str(x) for x in value.keys()]
    return list(fallback)


def key_name(label: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")


def normalized_channel(label: str) -> str:
    return label.strip().upper()


def is_permanent_bad_channel(label: str) -> bool:
    return normalized_channel(label) in PERMANENT_BAD_CHANNELS


def fmt(value: Any) -> str:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(x):
        return "NaN"
    if abs(x) >= 10000 or (0 < abs(x) < 0.0001):
        return f"{x:.6e}"
    return f"{x:.6g}"


def safe_subtract(a: Any, b: Any) -> str:
    try:
        x = float(a)
        y = float(b)
    except (TypeError, ValueError):
        return "NaN"
    if not math.isfinite(x) or not math.isfinite(y):
        return "NaN"
    return fmt(x - y)


def dataset(f: h5py.File | H5ReadCache, path: str) -> np.ndarray | None:
    if path not in f:
        return None
    return f[path]


def read_value(
    f: h5py.File | H5ReadCache,
    path: str,
    prefix: Tuple[int, ...],
    suffix: Tuple[int, ...] = (),
) -> Any:
    dset = dataset(f, path)
    if dset is None:
        return np.nan
    try:
        return dset[prefix + suffix]
    except (IndexError, ValueError):
        return np.nan


def read_part_value(
    f: h5py.File | H5ReadCache,
    path: str,
    ref: SegmentRef,
    part_index: int,
    suffix: Tuple[int, ...] = (),
) -> Any:
    if isinstance(f, H5ReadCache):
        values = f.segment_array(path, ref)
        if values is None:
            return np.nan
        try:
            return values[(part_index,) + suffix]
        except (IndexError, ValueError):
            return np.nan
    return read_value(f, path, (ref.trial_index, ref.segment_index, part_index), suffix)


def channel_index(channels: Mapping[str, int], label: str) -> int | None:
    direct = channels.get(label.upper())
    if direct is not None:
        return direct
    actual = CHANNEL_ALIASES.get(label, label).upper()
    return channels.get(actual)


def trial_bad_channel_mask(
    f: h5py.File | H5ReadCache,
    ref: SegmentRef,
    channels: Sequence[str],
    include_permanent: bool = True,
) -> np.ndarray:
    if isinstance(f, H5ReadCache):
        cached_mask = f.trial_array("meta/bad_channel_mask", ref.trial_index)
    else:
        mask_dset = dataset(f, "meta/bad_channel_mask")
        cached_mask = None if mask_dset is None else np.asarray(mask_dset[ref.trial_index])
    if cached_mask is None:
        bad_mask = np.zeros(len(channels), dtype=bool)
    else:
        bad_mask = np.asarray(cached_mask, dtype=bool)
        if bad_mask.shape[0] != len(channels):
            fixed = np.zeros(len(channels), dtype=bool)
            n = min(bad_mask.shape[0], len(channels))
            fixed[:n] = bad_mask[:n]
            bad_mask = fixed
    if include_permanent:
        bad_mask = bad_mask | np.asarray(
            [is_permanent_bad_channel(ch) for ch in channels], dtype=bool
        )
    return bad_mask


def valid_channel_index(
    f: h5py.File,
    ref: SegmentRef,
    channels: Sequence[str],
    channel_map: Mapping[str, int],
    label: str,
) -> int | None:
    idx = channel_index(channel_map, label)
    if idx is None:
        return None
    bad_mask = trial_bad_channel_mask(f, ref, channels, include_permanent=True)
    return None if bool(bad_mask[idx]) else idx


def pair_has_bad_channel(
    f: h5py.File,
    ref: SegmentRef,
    channels: Sequence[str],
    pair: str,
) -> bool:
    channel_map = {ch.upper(): i for i, ch in enumerate(channels)}
    labels = [part.strip() for part in pair.split("-")]
    return any(
        valid_channel_index(f, ref, channels, channel_map, label) is None
        for label in labels
    )


def pair_index(pair_names: Mapping[str, int], label: str) -> int | None:
    if any(is_permanent_bad_channel(part) for part in label.split("-")):
        return None
    direct = pair_names.get(label.upper())
    if direct is not None:
        return direct
    actual = PAIR_ALIASES.get(label, label).upper()
    return pair_names.get(actual)


def source_available(
    f: h5py.File | H5ReadCache, ref: SegmentRef, part_index: int
) -> str:
    if isinstance(f, H5ReadCache):
        segment_values = f.segment_array("source_network/abs_power", ref)
        if segment_values is None:
            return "false"
        values = segment_values[part_index]
    else:
        arr = dataset(f, "source_network/abs_power")
        if arr is None:
            return "false"
        values = arr[ref.trial_index, ref.segment_index, part_index]
    if values is None:
        return "false"
    return "true" if np.isfinite(values).any() else "false"


def qc_context(
    f: h5py.File | H5ReadCache, ref: SegmentRef, channels: Sequence[str]
) -> Dict[str, str]:
    raw_bad_mask = trial_bad_channel_mask(f, ref, channels, include_permanent=False)
    fixed_bad_mask = np.asarray([is_permanent_bad_channel(ch) for ch in channels], dtype=bool)
    qc_mask = ~fixed_bad_mask
    if isinstance(f, H5ReadCache):
        bad_trial_value = f.trial_array("meta/bad_trial_mask", ref.trial_index)
        trial_bad = bool(bad_trial_value) if bad_trial_value is not None else False
    else:
        bad_trial_dset = dataset(f, "meta/bad_trial_mask")
        trial_bad = (
            bool(bad_trial_dset[ref.trial_index]) if bad_trial_dset is not None else False
        )
    bad_mask_for_qc = raw_bad_mask & qc_mask
    bad = [ch for ch, is_bad in zip(channels, bad_mask_for_qc) if is_bad]
    valid = [
        ch
        for ch, is_qc_channel, is_bad in zip(channels, qc_mask, raw_bad_mask)
        if is_qc_channel and not is_bad
    ]
    ratio = float(np.count_nonzero(bad_mask_for_qc)) / max(int(np.count_nonzero(qc_mask)), 1)
    return {
        "True/False": "True" if trial_bad else "False",
        "trial_bad": "True" if trial_bad else "False",
        "bad_channel_ratio": fmt(ratio),
        "valid_channels": ", ".join(valid) if valid else "none",
        "bad_channels": ", ".join(bad) if bad else "none",
    }


def build_part_context(
    f: h5py.File,
    ref: SegmentRef,
    part_index: int,
    channels: Sequence[str],
    band_names: Sequence[str],
    roi_names: Sequence[str],
    pair_names: Sequence[str],
) -> Dict[str, str]:
    channel_map = {ch.upper(): i for i, ch in enumerate(channels)}
    band_map = {name: i for i, name in enumerate(band_names)}
    roi_map = {name: i for i, name in enumerate(roi_names)}
    pair_map = {name.upper(): i for i, name in enumerate(pair_names)}
    ctx = qc_context(f, ref, channels)

    for label in CONTEXT_CHANNELS:
        prefix = key_name(label)
        ch_idx = valid_channel_index(f, ref, channels, channel_map, label)
        for band in BANDS:
            band_idx = band_map.get(band)
            suffix = (ch_idx, band_idx) if ch_idx is not None and band_idx is not None else ()
            ctx[f"{prefix}_{band}_abs"] = fmt(
                read_part_value(f, "bandpower/abs_power", ref, part_index, suffix)
                if suffix
                else np.nan
            )
            ctx[f"{prefix}_{band}_rel"] = fmt(
                read_part_value(f, "bandpower/rel_power", ref, part_index, suffix)
                if suffix
                else np.nan
            )
        ctx[f"{prefix}_total_power"] = fmt(
            read_part_value(f, "bandpower/total_power_0p5_45", ref, part_index, (ch_idx,))
            if ch_idx is not None
            else np.nan
        )
        for metric in (
            "ratio_theta_beta",
            "ratio_alpha_theta",
            "iaf_hz",
            "spectral_entropy",
            "sef_hz",
            "aperiodic_exponent",
            "alpha_oscillatory_power",
        ):
            ctx[f"{prefix}_{metric}"] = fmt(
                read_part_value(f, f"spectral_advanced/{metric}", ref, part_index, (ch_idx,))
                if ch_idx is not None
                else np.nan
            )
        for metric in (
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
        ):
            ctx[f"{prefix}_{metric}"] = fmt(
                read_part_value(f, f"time_domain/{metric}", ref, part_index, (ch_idx,))
                if ch_idx is not None
                else np.nan
            )

    for roi in ROIS:
        roi_idx = roi_map.get(roi)
        for band in BANDS:
            band_idx = band_map.get(band)
            suffix = (roi_idx, band_idx) if roi_idx is not None and band_idx is not None else ()
            ctx[f"{roi}_{band}_abs"] = fmt(
                read_part_value(f, "spatial/roi_abs_power", ref, part_index, suffix)
                if suffix
                else np.nan
            )
            ctx[f"{roi}_{band}_rel"] = fmt(
                read_part_value(f, "spatial/roi_rel_power", ref, part_index, suffix)
                if suffix
                else np.nan
            )
        ctx[f"{roi}_n_good"] = fmt(
            read_part_value(f, "spatial/roi_n_good", ref, part_index, (roi_idx,))
            if roi_idx is not None
            else np.nan
        )

    for pair in PAIRS:
        pair_idx = None if pair_has_bad_channel(f, ref, channels, pair) else pair_index(pair_map, pair)
        pair_key = key_name(pair)
        for band in BANDS:
            band_idx = band_map.get(band)
            suffix = (pair_idx, band_idx) if pair_idx is not None and band_idx is not None else ()
            ctx[f"{pair_key}_{band}_asymmetry_log"] = fmt(
                read_part_value(f, "spatial/asymmetry_log", ref, part_index, suffix)
                if suffix
                else np.nan
            )
            ctx[f"{pair_key}_{band}_lateralization_index"] = fmt(
                read_part_value(f, "spatial/lateralization_index", ref, part_index, suffix)
                if suffix
                else np.nan
            )

    for left, right in ROI_RATIO_PAIRS:
        left_idx = roi_map.get(ROI_SHORT[left])
        right_idx = roi_map.get(ROI_SHORT[right])
        for band in BANDS:
            band_idx = band_map.get(band)
            if left_idx is None or right_idx is None or band_idx is None:
                value = np.nan
            else:
                num = read_part_value(f, "spatial/roi_abs_power", ref, part_index, (left_idx, band_idx))
                den = read_part_value(f, "spatial/roi_abs_power", ref, part_index, (right_idx, band_idx))
                value = float(num) / float(den) if np.isfinite(num) and np.isfinite(den) and den != 0 else np.nan
            ctx[f"ratio_{band}_{left}_{right}"] = fmt(value)

    for net_idx, (_, net_key) in enumerate(SOURCE_NETWORKS):
        for band in BANDS:
            band_idx = band_map.get(band)
            suffix = (net_idx, band_idx) if band_idx is not None else ()
            ctx[f"{net_key}_{band}_abs"] = fmt(
                read_part_value(f, "source_network/abs_power", ref, part_index, suffix)
                if suffix
                else np.nan
            )
            ctx[f"{net_key}_{band}_rel"] = fmt(
                read_part_value(f, "source_network/rel_power", ref, part_index, suffix)
                if suffix
                else np.nan
            )
        ctx[f"{net_key}_total_power"] = fmt(
            read_part_value(f, "source_network/total_power_0p5_45", ref, part_index, (net_idx,))
        )
    ctx["source_available"] = source_available(f, ref, part_index)
    return ctx


def render_template(template: str, context: Mapping[str, str]) -> str:
    rendered = template.format_map(SafeDict(context))
    return re.sub(r"\{[A-Za-z0-9_]+\}", "NaN", rendered)


# v2 already has one QC block at the top; strip title + QC from embedded v1 parts.
_EMBEDDED_V1_PREAMBLE_RE = re.compile(
    r"\A(?:#[^\n]*\n+)?"
    r"##\s*\*\*Quality control\*\*\s*\n+"
    r"(?:[ \t]*-[^\n]*\n)+"
    r"\n*",
)


def strip_embedded_v1_preamble(markdown: str) -> str:
    """Drop the leading title/QC block when nesting a v1 summary into v2."""
    return _EMBEDDED_V1_PREAMBLE_RE.sub("", markdown, count=1).lstrip("\n")


def build_change_context(
    f: h5py.File,
    ref: SegmentRef,
    channels: Sequence[str],
    band_names: Sequence[str],
    roi_names: Sequence[str],
    pair_names: Sequence[str],
) -> Dict[str, str]:
    channel_map = {ch.upper(): i for i, ch in enumerate(channels)}
    band_map = {name: i for i, name in enumerate(band_names)}
    roi_map = {name: i for i, name in enumerate(roi_names)}
    pair_map = {name.upper(): i for i, name in enumerate(pair_names)}
    ctx = qc_context(f, ref, channels)

    for left, right in ROI_RATIO_PAIRS:
        left_idx = roi_map.get(ROI_SHORT[left])
        right_idx = roi_map.get(ROI_SHORT[right])
        for band in BANDS:
            band_idx = band_map.get(band)
            values = []
            for part in range(3):
                if left_idx is None or right_idx is None or band_idx is None:
                    values.append(np.nan)
                    continue
                num = read_part_value(f, "spatial/roi_abs_power", ref, part, (left_idx, band_idx))
                den = read_part_value(f, "spatial/roi_abs_power", ref, part, (right_idx, band_idx))
                values.append(float(num) / float(den) if np.isfinite(num) and np.isfinite(den) and den != 0 else np.nan)
            ctx[f"{band}_middle_early_{left}_{right}"] = safe_subtract(values[1], values[0])
            ctx[f"{band}_last_middle_{left}_{right}"] = safe_subtract(values[2], values[1])
    return ctx


def make_channel_change_context(
    f: h5py.File,
    ref: SegmentRef,
    channel_label: str,
    channels: Sequence[str],
    band_names: Sequence[str],
) -> Dict[str, str]:
    channel_map = {ch.upper(): i for i, ch in enumerate(channels)}
    band_map = {name: i for i, name in enumerate(band_names)}
    ch_idx = valid_channel_index(f, ref, channels, channel_map, channel_label)
    ctx = {"channel": channel_label}
    for band in BANDS:
        band_idx = band_map.get(band)
        if ch_idx is None or band_idx is None:
            abs_vals = rel_vals = [np.nan, np.nan, np.nan]
        else:
            abs_vals = [read_part_value(f, "bandpower/abs_power", ref, p, (ch_idx, band_idx)) for p in range(3)]
            rel_vals = [read_part_value(f, "bandpower/rel_power", ref, p, (ch_idx, band_idx)) for p in range(3)]
        ctx[f"{band}_abs_me"] = safe_subtract(abs_vals[1], abs_vals[0])
        ctx[f"{band}_abs_lm"] = safe_subtract(abs_vals[2], abs_vals[1])
        ctx[f"{band}_rel_me"] = safe_subtract(rel_vals[1], rel_vals[0])
        ctx[f"{band}_rel_lm"] = safe_subtract(rel_vals[2], rel_vals[1])
    if ch_idx is None:
        total_vals = [np.nan, np.nan, np.nan]
    else:
        total_vals = [read_part_value(f, "bandpower/total_power_0p5_45", ref, p, (ch_idx,)) for p in range(3)]
    ctx["total_power_me"] = safe_subtract(total_vals[1], total_vals[0])
    ctx["total_power_lm"] = safe_subtract(total_vals[2], total_vals[1])
    for short, metric in SPECTRAL_CHANGE_KEYS.items():
        vals = [
            read_part_value(f, f"spectral_advanced/{metric}", ref, p, (ch_idx,))
            if ch_idx is not None
            else np.nan
            for p in range(3)
        ]
        ctx[f"{short}_me"] = safe_subtract(vals[1], vals[0])
        ctx[f"{short}_lm"] = safe_subtract(vals[2], vals[1])
    for short, metric in TIME_CHANGE_KEYS.items():
        vals = [
            read_part_value(f, f"time_domain/{metric}", ref, p, (ch_idx,))
            if ch_idx is not None
            else np.nan
            for p in range(3)
        ]
        ctx[f"{short}_me"] = safe_subtract(vals[1], vals[0])
        ctx[f"{short}_lm"] = safe_subtract(vals[2], vals[1])
    return ctx


def make_roi_change_context(
    f: h5py.File,
    ref: SegmentRef,
    roi: str,
    band_names: Sequence[str],
    roi_names: Sequence[str],
) -> Dict[str, str]:
    roi_map = {name: i for i, name in enumerate(roi_names)}
    band_map = {name: i for i, name in enumerate(band_names)}
    roi_idx = roi_map.get(roi)
    ctx = {"roi": roi}
    for band in BANDS:
        band_idx = band_map.get(band)
        for kind, path in (("abs", "spatial/roi_abs_power"), ("rel", "spatial/roi_rel_power")):
            vals = [
                read_part_value(f, path, ref, p, (roi_idx, band_idx))
                if roi_idx is not None and band_idx is not None
                else np.nan
                for p in range(3)
            ]
            ctx[f"roi_{band}_{kind}_me"] = safe_subtract(vals[1], vals[0])
            ctx[f"roi_{band}_{kind}_lm"] = safe_subtract(vals[2], vals[1])
    return ctx


def make_pair_change_context(
    f: h5py.File,
    ref: SegmentRef,
    pair: str,
    channels: Sequence[str],
    band_names: Sequence[str],
    pair_names: Sequence[str],
) -> Dict[str, str]:
    pair_map = {name.upper(): i for i, name in enumerate(pair_names)}
    band_map = {name: i for i, name in enumerate(band_names)}
    pair_idx = None if pair_has_bad_channel(f, ref, channels, pair) else pair_index(pair_map, pair)
    ctx = {"pair": pair}
    for band in BANDS:
        band_idx = band_map.get(band)
        for short, path in (("dasym", "spatial/asymmetry_log"), ("dlat", "spatial/lateralization_index")):
            vals = [
                read_part_value(f, path, ref, p, (pair_idx, band_idx))
                if pair_idx is not None and band_idx is not None
                else np.nan
                for p in range(3)
            ]
            ctx[f"{short}_{band}_me"] = safe_subtract(vals[1], vals[0])
            ctx[f"{short}_{band}_lm"] = safe_subtract(vals[2], vals[1])
    return ctx


def make_network_change_context(
    f: h5py.File,
    ref: SegmentRef,
    network_idx: int,
    network_name: str,
    band_names: Sequence[str],
) -> Dict[str, str]:
    band_map = {name: i for i, name in enumerate(band_names)}
    ctx = {"network": network_name}
    for band in BANDS:
        band_idx = band_map.get(band)
        for kind, path in (("abs", "source_network/abs_power"), ("rel", "source_network/rel_power")):
            vals = [
                read_part_value(f, path, ref, p, (network_idx, band_idx))
                if band_idx is not None
                else np.nan
                for p in range(3)
            ]
            ctx[f"src_{band}_{kind}_me"] = safe_subtract(vals[1], vals[0])
            ctx[f"src_{band}_{kind}_lm"] = safe_subtract(vals[2], vals[1])
    totals = [read_part_value(f, "source_network/total_power_0p5_45", ref, p, (network_idx,)) for p in range(3)]
    ctx["src_total_me"] = safe_subtract(totals[1], totals[0])
    ctx["src_total_lm"] = safe_subtract(totals[2], totals[1])
    return ctx


def render_v2(
    template: str,
    context: Mapping[str, str],
    row_contexts: Mapping[str, Callable[[str], Dict[str, str]]],
) -> str:
    rendered_lines: List[str] = []
    row_cache: Dict[Tuple[str, ...], Dict[str, str]] = {}

    def cached_row(kind: str, *args: str) -> Dict[str, str]:
        key = (kind,) + args
        if key not in row_cache:
            row_cache[key] = row_contexts[kind](*args)
        return row_cache[key]

    for line in template.splitlines():
        if "{channel}" in line:
            for channel in TEMPLATE_CHANNELS:
                rendered_lines.append(
                    render_template(line, {**context, **cached_row("channel", channel)})
                )
        elif "{roi}" in line:
            for roi in ROIS:
                rendered_lines.append(
                    render_template(line, {**context, **cached_row("roi", roi)})
                )
        elif "{pair}" in line:
            for pair in PAIRS:
                rendered_lines.append(
                    render_template(line, {**context, **cached_row("pair", pair)})
                )
        elif "{network}" in line:
            for idx, (network, _) in enumerate(SOURCE_NETWORKS):
                rendered_lines.append(
                    render_template(
                        line,
                        {
                            **context,
                            **cached_row("network", str(idx), network),
                        },
                    )
                )
        else:
            rendered_lines.append(render_template(line, context))
    return "\n".join(rendered_lines) + "\n"


def batch_dir_name(start: int, end: int) -> str:
    """Folder name for H5 index range [start, end) using 1-based display end."""
    return f"{start}-{end}"


def iter_file_batches(
    paths: Sequence[Path],
    batch_size: int,
    batch_index: int | None = None,
) -> List[Tuple[int, int, List[Path]]]:
    if batch_size <= 0:
        raise ValueError(f"batch_size must be > 0, got {batch_size}")
    batches: List[Tuple[int, int, List[Path]]] = []
    for start in range(0, len(paths), batch_size):
        end = min(start + batch_size, len(paths))
        batches.append((start, end, list(paths[start:end])))
    if batch_index is None:
        return batches
    if batch_index < 0 or batch_index >= len(batches):
        raise ValueError(
            f"batch_index {batch_index} out of range for {len(batches)} batches "
            f"(batch_size={batch_size}, n_files={len(paths)})"
        )
    return [batches[batch_index]]


def collect_segments(
    paths: Iterable[Path],
    limit: int | None = None,
    trial_filter: int | None = None,
    segment_filter: int | None = None,
) -> Tuple[List[SegmentRef], List[Dict[str, Any]]]:
    """Return good segment refs to summarize and a list of bad-segment records."""
    refs: List[SegmentRef] = []
    bad_records: List[Dict[str, Any]] = []
    for path in paths:
        with h5py.File(path, "r") as f:
            valid = np.asarray(f["meta/segment_valid_mask"][:], dtype=bool)
            bad_seg_dset = dataset(f, "meta/bad_segment_mask")
            bad_segments = (
                np.asarray(bad_seg_dset[:], dtype=bool)
                if bad_seg_dset is not None
                else np.zeros_like(valid, dtype=bool)
            )
            bad_trial_dset = dataset(f, "meta/bad_trial_mask")
            bad_trials = (
                np.asarray(bad_trial_dset[:], dtype=bool)
                if bad_trial_dset is not None
                else np.zeros(valid.shape[0], dtype=bool)
            )
            trial_ids = decode_array(f["meta/trial_ids"][:])
            starts_dset = dataset(f, "meta/segment_start_sec")
            segment_starts = (
                np.asarray(starts_dset[:], dtype=np.float32)
                if starts_dset is not None
                else None
            )

            n_trials = valid.shape[0]
            n_segments = valid.shape[1] if valid.ndim == 2 else 0
            for trial_index in range(n_trials):
                trial_id = (
                    trial_ids[trial_index]
                    if trial_index < len(trial_ids)
                    else f"trial{trial_index}"
                )
                trial_is_bad = trial_index < bad_trials.shape[0] and bool(bad_trials[trial_index])
                for segment_index in range(n_segments):
                    if trial_filter is not None and trial_index != trial_filter:
                        continue
                    if segment_filter is not None and segment_index != segment_filter:
                        continue

                    is_bad_segment = (
                        trial_index < bad_segments.shape[0]
                        and segment_index < bad_segments.shape[1]
                        and bool(bad_segments[trial_index, segment_index])
                    )
                    is_valid = bool(valid[trial_index, segment_index])
                    if not is_bad_segment and not is_valid:
                        continue

                    start_sec: float | None = None
                    if (
                        segment_starts is not None
                        and trial_index < segment_starts.shape[0]
                        and segment_index < segment_starts.shape[1]
                    ):
                        value = float(segment_starts[trial_index, segment_index])
                        start_sec = None if math.isnan(value) else value

                    if is_bad_segment or trial_is_bad:
                        bad_records.append(
                            {
                                "h5": str(path),
                                "trial_index": trial_index,
                                "trial_id": trial_id,
                                "segment_index": segment_index,
                                "segment_start_sec": start_sec,
                                "reason": (
                                    "bad_trial"
                                    if trial_is_bad and not is_bad_segment
                                    else "bad_segment"
                                ),
                            }
                        )
                        continue

                    if limit is None or len(refs) < limit:
                        refs.append(SegmentRef(path, trial_index, segment_index))
    return refs, bad_records


def clean_outputs(out_dir: Path) -> None:
    for subdir in ("template_v1_parts", "template_v2_segments"):
        path = out_dir / subdir
        if not path.exists():
            continue
        for md_path in path.glob("*.md"):
            md_path.unlink()
    for name in ("manifest.json", "bad_segments.json"):
        path = out_dir / name
        if path.exists():
            path.unlink()


def output_stem(
    f: h5py.File | H5ReadCache,
    ref: SegmentRef,
    trial_ids: Sequence[str] | None = None,
) -> str:
    if trial_ids is None:
        trial_ids = decode_array(f["meta/trial_ids"][:])
    trial_id = trial_ids[ref.trial_index] if ref.trial_index < len(trial_ids) else f"trial{ref.trial_index}"
    return f"{ref.h5_path.stem}__{trial_id}__seg{ref.segment_index:04d}"


def atomic_write_text(path: Path, text: str) -> None:
    """Publish a complete file in one rename so interrupted runs are resumable."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise


def file_is_complete(path: Path) -> bool:
    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def segment_output_paths(
    v1_dir: Path,
    v2_dir: Path,
    stem: str,
    part_names: Sequence[str],
) -> Tuple[List[Path], Path]:
    part_paths = [v1_dir / f"{stem}__{part_name}.md" for part_name in part_names]
    return part_paths, v2_dir / f"{stem}.md"


def render_batch(
    paths: Sequence[Path],
    out_dir: Path,
    template_v1: str,
    template_v2: str,
    *,
    limit: int | None,
    trial_filter: int | None,
    segment_filter: int | None,
    keep_existing: bool,
    resume: bool,
    input_glob: str,
    batch_meta: Mapping[str, Any],
) -> Dict[str, Any]:
    v1_dir = out_dir / "template_v1_parts"
    v2_dir = out_dir / "template_v2_segments"
    if not keep_existing and not resume:
        clean_outputs(out_dir)
    v1_dir.mkdir(parents=True, exist_ok=True)
    v2_dir.mkdir(parents=True, exist_ok=True)

    refs, bad_records = collect_segments(
        paths,
        limit=limit,
        trial_filter=trial_filter,
        segment_filter=segment_filter,
    )
    records: List[Dict[str, Any]] = []
    n_rendered = 0
    n_skipped_existing = 0
    render_started = time.monotonic()
    refs_by_path: Dict[Path, List[SegmentRef]] = defaultdict(list)
    for ref in refs:
        refs_by_path[ref.h5_path].append(ref)

    for h5_path, file_refs in refs_by_path.items():
        with h5py.File(h5_path, "r") as raw_f:
            f = H5ReadCache(raw_f)
            channels = decode_array(f["meta/channels"][:])
            band_names = load_json_attr(f.attrs, "bands_json", BANDS)
            roi_names = load_json_attr(f.attrs, "spatial_roi_names_json", ROIS)
            pair_names = load_json_attr(f.attrs, "spatial_pair_names_json", PAIRS)
            part_names = decode_array(f["meta/segment_part_names"][:])
            trial_ids = decode_array(f["meta/trial_ids"][:])

            for ref in file_refs:
                stem = output_stem(f, ref, trial_ids)
                part_paths, segment_path = segment_output_paths(
                    v1_dir, v2_dir, stem, part_names
                )
                record = {
                    "h5": str(ref.h5_path),
                    "trial_index": ref.trial_index,
                    "segment_index": ref.segment_index,
                    "template_v2": str(segment_path),
                    "template_v1_parts": [str(path) for path in part_paths],
                }
                if resume and all(
                    file_is_complete(path) for path in [*part_paths, segment_path]
                ):
                    record["status"] = "skipped_existing"
                    records.append(record)
                    n_skipped_existing += 1
                    continue

                f.begin_segment(ref)
                part_summaries: Dict[str, str] = {}
                rendered_parts: List[str] = []
                for part_index, part_name in enumerate(part_names):
                    part_context = build_part_context(
                        f, ref, part_index, channels, band_names, roi_names, pair_names
                    )
                    rendered = render_template(template_v1, part_context)
                    rendered_parts.append(rendered)
                    # Standalone v1 keeps QC; embed into v2 without repeating it.
                    part_summaries[part_name] = strip_embedded_v1_preamble(rendered)

                change_context = build_change_context(
                    f, ref, channels, band_names, roi_names, pair_names
                )
                v2_context = {
                    **qc_context(f, ref, channels),
                    **change_context,
                    "early_segment_eeg_feature_summaryv1": part_summaries.get(
                        "early", "NaN"
                    ),
                    "middle_segment_eeg_feature_summaryv1": part_summaries.get(
                        "middle", "NaN"
                    ),
                    "last_segment_eeg_feature_summaryv1": part_summaries.get(
                        "last", "NaN"
                    ),
                }
                row_contexts: Dict[str, Callable[..., Dict[str, str]]] = {
                    "channel": lambda ch, f=f, ref=ref, channels=channels, band_names=band_names: make_channel_change_context(
                        f, ref, ch, channels, band_names
                    ),
                    "roi": lambda roi, f=f, ref=ref, band_names=band_names, roi_names=roi_names: make_roi_change_context(
                        f, ref, roi, band_names, roi_names
                    ),
                    "pair": lambda pair, f=f, ref=ref, channels=channels, band_names=band_names, pair_names=pair_names: make_pair_change_context(
                        f, ref, pair, channels, band_names, pair_names
                    ),
                    "network": lambda idx, name, f=f, ref=ref, band_names=band_names: make_network_change_context(
                        f, ref, int(idx), name, band_names
                    ),
                }
                rendered_v2 = render_v2(template_v2, v2_context, row_contexts)

                # Publish v2 last: its presence plus all v1 files marks completion.
                for part_path, rendered in zip(part_paths, rendered_parts):
                    atomic_write_text(part_path, rendered)
                atomic_write_text(segment_path, rendered_v2)

                record["status"] = "rendered"
                records.append(record)
                n_rendered += 1
                if n_rendered % 100 == 0:
                    elapsed = time.monotonic() - render_started
                    print(
                        json.dumps(
                            {
                                "batch_dir": str(out_dir),
                                "rendered": n_rendered,
                                "skipped_existing": n_skipped_existing,
                                "segments_per_sec": n_rendered / max(elapsed, 1e-9),
                            }
                        ),
                        flush=True,
                    )

    bad_path = out_dir / "bad_segments.json"
    atomic_write_text(
        bad_path,
        json.dumps(
            {
                **batch_meta,
                "n_bad_segments": len(bad_records),
                "bad_segments": bad_records,
            },
            indent=2,
            ensure_ascii=False,
        ),
    )

    manifest = {
        **batch_meta,
        "input_glob": input_glob,
        "n_h5": len(paths),
        "n_segments": len(records),
        "n_rendered": n_rendered,
        "n_skipped_existing": n_skipped_existing,
        "n_bad_segments": len(bad_records),
        "bad_segments_path": str(bad_path),
        "segments": records,
    }
    manifest_path = out_dir / "manifest.json"
    atomic_write_text(
        manifest_path, json.dumps(manifest, indent=2, ensure_ascii=False)
    )
    return {
        "batch_dir": str(out_dir),
        "manifest": str(manifest_path),
        "bad_segments": str(bad_path),
        "n_h5": len(paths),
        "n_segments": len(records),
        "n_rendered": n_rendered,
        "n_skipped_existing": n_skipped_existing,
        "n_bad_segments": len(bad_records),
    }


def main() -> None:
    args = parse_args()
    template_v1 = Path(args.template_v1).read_text(encoding="utf-8")
    template_v2 = Path(args.template_v2).read_text(encoding="utf-8")
    root_out = Path(args.output_dir)

    h5_paths = [Path(p) for p in sorted(glob.glob(args.input_glob))]
    if args.max_files is not None:
        h5_paths = h5_paths[: max(args.max_files, 0)]
    if not h5_paths:
        raise SystemExit(f"No H5 files matched: {args.input_glob}")

    batches = iter_file_batches(h5_paths, args.batch_size, args.batch_index)
    summaries: List[Dict[str, Any]] = []
    for start, end, paths in batches:
        batch_name = batch_dir_name(start, end)
        out_dir = root_out / batch_name
        batch_meta = {
            "batch_name": batch_name,
            "h5_index_start": start,
            "h5_index_end": end,
            "h5_files": [str(p) for p in paths],
        }
        summary = render_batch(
            paths,
            out_dir,
            template_v1,
            template_v2,
            limit=args.limit,
            trial_filter=args.trial_index,
            segment_filter=args.segment_index,
            keep_existing=args.keep_existing,
            resume=args.resume,
            input_glob=args.input_glob,
            batch_meta=batch_meta,
        )
        summaries.append(summary)
        print(json.dumps(summary, indent=2))

    if len(summaries) > 1:
        print(
            json.dumps(
                {
                    "n_batches": len(summaries),
                    "n_segments": sum(item["n_segments"] for item in summaries),
                    "n_rendered": sum(item["n_rendered"] for item in summaries),
                    "n_skipped_existing": sum(
                        item["n_skipped_existing"] for item in summaries
                    ),
                    "n_bad_segments": sum(item["n_bad_segments"] for item in summaries),
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
