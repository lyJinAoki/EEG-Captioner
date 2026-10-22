"""I/O helpers for mirrored TUEG H5 feature extraction."""

from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Iterator, List, Tuple

import h5py
import numpy as np


@dataclass
class InputMeta:
    input_path: str
    sample_rate: float
    all_channels: List[str]
    eeg_indices: np.ndarray
    eeg_channels: List[str]
    trial_lengths: np.ndarray
    trial_ids: List[str]
    source_paths: List[str]


def list_input_h5_files(input_root: str, max_files: int | None = None) -> List[str]:
    """List input H5 files in deterministic filename order."""
    pattern = os.path.join(input_root, "*.h5")
    files = sorted(glob.glob(pattern))
    if max_files is not None:
        return files[: int(max_files)]
    return files


def build_output_path(input_path: str, input_root: str, output_root: str) -> str:
    """Mirror input file path under output root."""
    rel = os.path.relpath(input_path, input_root)
    out_path = os.path.join(output_root, rel)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    return out_path


def _decode_array(values: np.ndarray, fallback_prefix: str = "item") -> List[str]:
    decoded: List[str] = []
    for i, value in enumerate(values):
        if isinstance(value, bytes):
            decoded.append(value.decode("utf-8", errors="replace"))
        else:
            decoded.append(str(value))
        if decoded[-1] == "":
            decoded[-1] = f"{fallback_prefix}_{i}"
    return decoded


def read_input_meta(input_path: str, use_labram_channels: bool = True) -> InputMeta:
    """Read required metadata from one preprocessed subject H5 file."""
    with h5py.File(input_path, "r") as f:
        channels = _decode_array(f["channels"][:], fallback_prefix="ch")
        trial_lengths = np.asarray(f["trial_lengths_in_samples"][:], dtype=np.int32)
        n_trials = int(f["data"].shape[0])
        sample_rate = float(f.attrs.get("sample_rate", 200.0))

        if use_labram_channels and "labram_channel_indices" in f:
            eeg_indices = np.asarray(f["labram_channel_indices"][:], dtype=np.int32)
        else:
            eeg_indices = np.arange(len(channels), dtype=np.int32)

        eeg_indices = eeg_indices[(eeg_indices >= 0) & (eeg_indices < len(channels))]
        eeg_indices = np.unique(eeg_indices).astype(np.int32)
        eeg_channels = [channels[i] for i in eeg_indices]

        if "trial_ids" in f:
            trial_ids = _decode_array(f["trial_ids"][:], fallback_prefix="trial")
        else:
            trial_ids = [f"trial_{i}" for i in range(n_trials)]
        if "source_paths" in f:
            source_paths = _decode_array(f["source_paths"][:], fallback_prefix="source")
        else:
            source_paths = ["" for _ in range(n_trials)]

    if len(trial_ids) < n_trials:
        trial_ids.extend(f"trial_{i}" for i in range(len(trial_ids), n_trials))
    if len(source_paths) < n_trials:
        source_paths.extend("" for _ in range(len(source_paths), n_trials))

    return InputMeta(
        input_path=input_path,
        sample_rate=sample_rate,
        all_channels=channels,
        eeg_indices=eeg_indices,
        eeg_channels=eeg_channels,
        trial_lengths=trial_lengths,
        trial_ids=trial_ids[:n_trials],
        source_paths=source_paths[:n_trials],
    )


def iter_trials(input_meta: InputMeta) -> Iterator[Tuple[int, np.ndarray, int]]:
    """Yield (trial_index, signal[eeg_channels, samples], effective_length)."""
    with h5py.File(input_meta.input_path, "r") as f:
        dset = f["data"]
        max_samples = int(dset.shape[2])
        n_trials = int(dset.shape[0])
        for trial_idx in range(n_trials):
            length = int(input_meta.trial_lengths[trial_idx])
            if length <= 0 or length > max_samples:
                length = max_samples
            trial = np.asarray(
                dset[trial_idx, input_meta.eeg_indices, :length], dtype=np.float64
            )
            yield trial_idx, trial, length


def _to_h5_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False)
    return value


def list_existing_feature_groups(output_path: str) -> set[str]:
    """Return feature groups from a structurally valid existing output H5."""
    if not os.path.exists(output_path):
        return set()
    with h5py.File(output_path, "r") as f:
        if "meta" not in f:
            raise ValueError(f"Existing output has no meta group: {output_path}")
        meta = f["meta"]
        for required in ("channels", "trial_ids", "source_paths"):
            if required not in meta:
                raise ValueError(
                    f"Existing output is missing meta/{required}: {output_path}"
                )
        return {name for name in f.keys() if name != "meta"}


def read_existing_bad_channel_mask(output_path: str) -> np.ndarray | None:
    """Read a previously stored per-trial bad-channel mask, if present.

    Returns a ``(n_trials, n_channels)`` bool array, or ``None`` when the
    output file has no stored mask (e.g. it was created with bad-channel
    detection disabled). Used to reuse the mask when appending new feature
    groups without recomputing detection.
    """
    if not os.path.exists(output_path):
        return None
    with h5py.File(output_path, "r") as f:
        meta = f.get("meta")
        if meta is None or "bad_channel_mask" not in meta:
            return None
        return np.asarray(meta["bad_channel_mask"][:], dtype=bool)


def read_existing_unit_count(output_path: str) -> int | None:
    """Return number of stored analysis units (rows in meta/trial_ids)."""
    if not os.path.exists(output_path):
        return None
    with h5py.File(output_path, "r") as f:
        meta = f.get("meta")
        if meta is None or "trial_ids" not in meta:
            return None
        return int(len(meta["trial_ids"]))


def read_output_attr(output_path: str, key: str, default: Any = None) -> Any:
    """Read a top-level HDF5 attribute from an existing output file."""
    if not os.path.exists(output_path):
        return default
    with h5py.File(output_path, "r") as f:
        if key not in f.attrs:
            return default
        value = f.attrs[key]
        if isinstance(value, bytes):
            return value.decode("utf-8")
        if isinstance(value, np.bool_):
            return bool(value)
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            return float(value)
        return value


def write_feature_h5(
    output_path: str,
    feature_groups: Dict[str, Dict[str, np.ndarray]],
    channels: Iterable[str],
    trial_ids: Iterable[str],
    source_paths: Iterable[str],
    attrs: Dict[str, Any],
    bad_channel_mask: np.ndarray | None = None,
    bad_trial_mask: np.ndarray | None = None,
    segment_start_sec: np.ndarray | None = None,
    parent_trial_index: np.ndarray | None = None,
    segment_valid_mask: np.ndarray | None = None,
    bad_segment_mask: np.ndarray | None = None,
    segment_part_names: Iterable[str] | None = None,
    segment_part_start_sec: np.ndarray | None = None,
    segment_part_duration_sec: np.ndarray | None = None,
    append: bool = False,
) -> None:
    """Write one mirrored output H5 with grouped feature arrays."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    file_exists = os.path.exists(output_path)
    mode = "a" if append and file_exists else "w"

    with h5py.File(output_path, mode) as f:
        for group_name, datasets in feature_groups.items():
            if append and group_name in f:
                continue
            group = f.create_group(group_name)
            for dataset_name, array in datasets.items():
                group.create_dataset(
                    dataset_name,
                    data=array,
                    compression="gzip",
                    compression_opts=4,
                )

        write_meta = mode == "w" or "meta" not in f
        if write_meta:
            meta = f.create_group("meta") if "meta" not in f else f["meta"]
            if "channels" not in meta:
                meta.create_dataset("channels", data=np.array(list(channels), dtype="S"))
                meta.create_dataset("trial_ids", data=np.array(list(trial_ids), dtype="S"))
                meta.create_dataset("source_paths", data=np.array(list(source_paths), dtype="S"))
            if bad_channel_mask is not None and "bad_channel_mask" not in meta:
                meta.create_dataset(
                    "bad_channel_mask",
                    data=np.asarray(bad_channel_mask, dtype=np.bool_),
                    compression="gzip",
                    compression_opts=4,
                )
            if bad_trial_mask is not None and "bad_trial_mask" not in meta:
                meta.create_dataset(
                    "bad_trial_mask",
                    data=np.asarray(bad_trial_mask, dtype=np.bool_),
                    compression="gzip",
                    compression_opts=4,
                )
            if segment_start_sec is not None and "segment_start_sec" not in meta:
                meta.create_dataset(
                    "segment_start_sec",
                    data=np.asarray(segment_start_sec, dtype=np.float32),
                    compression="gzip",
                    compression_opts=4,
                )
            if parent_trial_index is not None and "parent_trial_index" not in meta:
                meta.create_dataset(
                    "parent_trial_index",
                    data=np.asarray(parent_trial_index, dtype=np.int32),
                    compression="gzip",
                    compression_opts=4,
                )
            if segment_valid_mask is not None and "segment_valid_mask" not in meta:
                meta.create_dataset(
                    "segment_valid_mask",
                    data=np.asarray(segment_valid_mask, dtype=np.bool_),
                    compression="gzip",
                    compression_opts=4,
                )
            if bad_segment_mask is not None and "bad_segment_mask" not in meta:
                meta.create_dataset(
                    "bad_segment_mask",
                    data=np.asarray(bad_segment_mask, dtype=np.bool_),
                    compression="gzip",
                    compression_opts=4,
                )
            if segment_part_names is not None and "segment_part_names" not in meta:
                meta.create_dataset(
                    "segment_part_names",
                    data=np.array(list(segment_part_names), dtype="S"),
                )
            if segment_part_start_sec is not None and "segment_part_start_sec" not in meta:
                meta.create_dataset(
                    "segment_part_start_sec",
                    data=np.asarray(segment_part_start_sec, dtype=np.float32),
                    compression="gzip",
                    compression_opts=4,
                )
            if segment_part_duration_sec is not None and "segment_part_duration_sec" not in meta:
                meta.create_dataset(
                    "segment_part_duration_sec",
                    data=np.asarray(segment_part_duration_sec, dtype=np.float32),
                    compression="gzip",
                    compression_opts=4,
                )

        for key, value in attrs.items():
            if value is not None:
                f.attrs[key] = _to_h5_value(value)
