#!/usr/bin/env python3
"""Run mirrored EEG feature extraction from preprocessed TUEG H5 inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import time
import traceback
from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import yaml

from eeg_captioner.features.bad_channel import bad_channel_preprocess, segment_exceeds_hard_peak
from eeg_captioner.features.feature_registry import FeatureRegistry, TrialContext
from eeg_captioner.features.features.bandpower import build_bandpower_plugin
from eeg_captioner.features.features.spatial import build_spatial_plugin
from eeg_captioner.features.features.source_network import build_source_network_plugin
from eeg_captioner.features.features.spectral_advanced import build_spectral_advanced_plugin
from eeg_captioner.features.features.time_domain import build_time_domain_plugin
from eeg_captioner.features.source_artifacts import YEO7_NETWORK_NAMES
from eeg_captioner.features.segmentation import SegmentSpec, iter_equal_parts, iter_segments
from eeg_captioner.features.io_tueg_h5 import (
    build_output_path,
    iter_trials,
    list_existing_feature_groups,
    list_input_h5_files,
    read_existing_bad_channel_mask,
    read_existing_unit_count,
    read_input_meta,
    read_output_attr,
    write_feature_h5,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="EEG feature extraction pipeline.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument(
        "--bad-channel",
        action="store_true",
        help="Enable bad-channel detection (overrides bad_channel.enabled in YAML).",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Override max_files from YAML.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files.",
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        default=1,
        help="Split the deterministic input list across this many workers.",
    )
    parser.add_argument(
        "--shard-index",
        type=int,
        default=0,
        help="Zero-based shard handled by this process.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=50,
        help="Print a JSON progress record every N files (default: 50).",
    )
    return parser.parse_args()


def apply_cli_overrides(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    if args.max_files is not None:
        config["max_files"] = args.max_files
    if args.overwrite:
        config["overwrite"] = True
    if args.bad_channel:
        bad_channel = dict(config.get("bad_channel") or {})
        bad_channel["enabled"] = True
        config["bad_channel"] = bad_channel
    config["num_shards"] = args.num_shards
    config["shard_index"] = args.shard_index
    config["progress_every"] = args.progress_every
    return config


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise ValueError("Config must be a YAML mapping.")
    return config


def select_input_shard(
    input_files: Sequence[str],
    num_shards: int,
    shard_index: int,
) -> List[str]:
    """Select one deterministic round-robin shard from a sorted input list."""
    if num_shards <= 0:
        raise ValueError(f"num_shards must be > 0, got {num_shards}")
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError(
            f"shard_index {shard_index} out of range for num_shards={num_shards}"
        )
    return list(input_files[shard_index::num_shards])


def shard_file_suffix(num_shards: int, shard_index: int) -> str:
    if num_shards == 1:
        return ""
    width = max(2, len(str(num_shards - 1)))
    return f".shard-{shard_index:0{width}d}-of-{num_shards:0{width}d}"


def atomic_write_json(path: str, payload: Mapping[str, Any]) -> None:
    """Atomically publish JSON so interrupted jobs never leave a partial manifest."""
    directory = os.path.dirname(path)
    os.makedirs(directory, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        prefix=f".{os.path.basename(path)}.",
        suffix=".tmp",
        dir=directory,
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
        os.replace(tmp_path, path)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except FileNotFoundError:
            pass
        raise


def _aggregate(
    accumulator: Dict[str, Dict[str, List[np.ndarray]]],
    outputs: Dict[str, Dict[str, np.ndarray]],
) -> None:
    for feature_name, datasets in outputs.items():
        feature_bucket = accumulator.setdefault(feature_name, {})
        for dataset_name, array in datasets.items():
            feature_bucket.setdefault(dataset_name, []).append(np.asarray(array))


def _stack_accumulator(
    accumulator: Dict[str, Dict[str, List[np.ndarray]]],
) -> Dict[str, Dict[str, np.ndarray]]:
    stacked: Dict[str, Dict[str, np.ndarray]] = {}
    for feature_name, datasets in accumulator.items():
        stacked[feature_name] = {}
        for dataset_name, arrays in datasets.items():
            stacked[feature_name][dataset_name] = np.stack(arrays, axis=0).astype(np.float32)
    return stacked


def _part_names_from_config(segmentation_cfg: Mapping[str, Any] | None) -> Tuple[str, ...] | None:
    if not segmentation_cfg:
        return None
    thirds_cfg = segmentation_cfg.get("thirds")
    if not isinstance(thirds_cfg, Mapping) or not bool(thirds_cfg.get("enabled", False)):
        return None
    names = tuple(str(name) for name in thirds_cfg.get("names", ["early", "middle", "last"]))
    if len(names) != 3:
        raise ValueError("segmentation.thirds.names must contain exactly three names.")
    if len(set(names)) != len(names):
        raise ValueError("segmentation.thirds.names must be unique.")
    return names


def _stack_hierarchical_records(
    records: Sequence[Tuple[int, int, int, Dict[str, Dict[str, np.ndarray]]]],
    n_trials: int,
    max_segments: int,
    n_parts: int,
) -> Dict[str, Dict[str, np.ndarray]]:
    stacked: Dict[str, Dict[str, np.ndarray]] = {}
    for trial_index, segment_index, part_index, outputs in records:
        for feature_name, datasets in outputs.items():
            feature_bucket = stacked.setdefault(feature_name, {})
            for dataset_name, array in datasets.items():
                arr = np.asarray(array, dtype=np.float32)
                if dataset_name not in feature_bucket:
                    shape = (n_trials, max_segments, n_parts) + arr.shape
                    feature_bucket[dataset_name] = np.full(shape, np.nan, dtype=np.float32)
                feature_bucket[dataset_name][trial_index, segment_index, part_index] = arr
    return stacked


def _valid_values(array: np.ndarray, valid_mask: np.ndarray | None) -> np.ndarray:
    if valid_mask is None or array.shape[: valid_mask.ndim] != valid_mask.shape:
        return array
    return array[valid_mask]


def _run_sanity_checks(
    feature_groups: Dict[str, Dict[str, np.ndarray]],
    warnings: List[str],
    valid_mask: np.ndarray | None = None,
) -> None:
    nan_ok = {"spectral_advanced", "spatial", "source_network"}
    for feature_name, datasets in feature_groups.items():
        for dataset_name, array in datasets.items():
            check_array = _valid_values(array, valid_mask)
            if not np.isfinite(check_array).all():
                if feature_name in nan_ok and np.isinf(check_array).sum() == 0:
                    n_nan = int(np.isnan(check_array).sum())
                    warnings.append(f"{feature_name}/{dataset_name} has {n_nan} NaN (bad channels).")
                else:
                    raise ValueError(f"{feature_name}/{dataset_name} contains NaN or Inf")

    bandpower = feature_groups.get("bandpower", {})
    rel = bandpower.get("rel_power")
    if rel is not None:
        rel_check = _valid_values(rel, valid_mask)
        if (rel_check < 0).any() or (rel_check > 1).any():
            warnings.append("rel_power has values outside [0, 1].")
        rel_sum = _valid_values(rel.sum(axis=-1), valid_mask)
        if (rel_sum > 1.05).any():
            warnings.append("rel_power band-sum exceeds expected tolerance in some channels.")


def _build_registry(features_cfg: Mapping[str, Mapping[str, Any]]) -> FeatureRegistry:
    registry = FeatureRegistry()
    if "bandpower" in features_cfg:
        registry.register("bandpower", build_bandpower_plugin(features_cfg["bandpower"]))
    if "time_domain" in features_cfg:
        registry.register("time_domain", build_time_domain_plugin(features_cfg["time_domain"]))
    if "spectral_advanced" in features_cfg:
        registry.register(
            "spectral_advanced", build_spectral_advanced_plugin(features_cfg["spectral_advanced"])
        )
    if "spatial" in features_cfg:
        registry.register("spatial", build_spatial_plugin(features_cfg["spatial"]))
    if "source_network" in features_cfg:
        registry.register(
            "source_network", build_source_network_plugin(features_cfg["source_network"])
        )
    return registry


def run_pipeline(config: Dict[str, Any]) -> Dict[str, Any]:
    input_root = str(config["input_root"])
    output_root = str(config["output_root"])
    max_files = config.get("max_files")
    overwrite = bool(config.get("overwrite", False))
    num_shards = int(config.get("num_shards", 1))
    shard_index = int(config.get("shard_index", 0))
    progress_every = int(config.get("progress_every", 50))
    if progress_every <= 0:
        raise ValueError(f"progress_every must be > 0, got {progress_every}")
    use_labram_channels = bool(config.get("use_labram_channels", True))
    pipeline_version = str(config.get("pipeline_version", "default_v1"))
    segmentation_cfg = config.get("segmentation")
    segment_spec = SegmentSpec.from_config(segmentation_cfg)
    segment_part_names = _part_names_from_config(segmentation_cfg)

    bad_channel_cfg = config.get("bad_channel", {}) or {}
    bad_channel_enabled = bool(bad_channel_cfg.get("enabled", False))
    segment_reject_cfg = dict(bad_channel_cfg.get("segment_reject") or {})
    segment_reject_enabled = bool(segment_reject_cfg.get("enabled", False))
    hard_peak_threshold_uv = float(segment_reject_cfg.get("hard_peak_threshold_uv", 100.0))

    features_cfg = config.get("features", {}) or {}
    if not features_cfg:
        raise ValueError("Config contains no features.")

    enabled_features = {
        feature_name: feature_cfg
        for feature_name, feature_cfg in features_cfg.items()
        if bool(feature_cfg.get("enabled", True))
    }
    if not enabled_features:
        raise ValueError("All configured features are disabled.")

    # Fail once before a long run instead of publishing missing source features.
    if "source_network" in enabled_features:
        from eeg_captioner.features.source_artifacts import get_source_artifacts
        get_source_artifacts(enabled_features["source_network"])

    os.makedirs(output_root, exist_ok=True)

    all_input_files = list_input_h5_files(input_root, max_files=max_files)
    if not all_input_files:
        raise ValueError(f"No preprocessed H5 inputs found under {input_root}")
    input_files = select_input_shard(all_input_files, num_shards, shard_index)
    input_list_hash = hashlib.sha256(
        "\n".join(all_input_files).encode("utf-8")
    ).hexdigest()
    suffix = shard_file_suffix(num_shards, shard_index)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    manifest_path = os.path.join(output_root, f"manifest{suffix}.json")
    progress_path = os.path.join(
        output_root,
        f"progress{suffix}.{run_id}.jsonl",
    )
    manifest: Dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "pipeline_version": pipeline_version,
        "input_root": input_root,
        "output_root": output_root,
        "max_files": max_files,
        "num_shards": num_shards,
        "shard_index": shard_index,
        "files_global_total": len(all_input_files),
        "files_shard_total": len(input_files),
        "input_list_sha256": input_list_hash,
        "progress_path": progress_path,
        "manifest_path": manifest_path,
        "segmentation": (
            {
                "enabled": True,
                "duration_sec": segment_spec.duration_sec,
                "overlap_sec": segment_spec.overlap_sec,
                "min_duration_sec": segment_spec.min_duration_sec,
                "thirds": (
                    {"enabled": True, "names": list(segment_part_names)}
                    if segment_part_names
                    else {"enabled": False}
                ),
            }
            if segment_spec
            else None
        ),
        "files_total": len(input_files),
        "files": [],
    }

    with open(progress_path, "x", encoding="utf-8"):
        pass

    run_started = time.monotonic()
    status_counts = {
        "ok": 0,
        "appended": 0,
        "skipped_existing": 0,
        "error": 0,
    }

    for file_number, input_path in enumerate(input_files, start=1):
        basename = os.path.basename(input_path)
        start = time.time()
        file_record: Dict[str, Any] = {
            "file": basename,
            "input_path": input_path,
            "status": "unknown",
            "warnings": [],
        }
        try:
            output_path = build_output_path(input_path, input_root, output_root)
            file_record["output_path"] = output_path

            if overwrite:
                pending_features = dict(enabled_features)
                append_mode = False
            elif os.path.exists(output_path):
                try:
                    existing_groups = list_existing_feature_groups(output_path)
                    all_segments_rejected = bool(
                        read_output_attr(output_path, "all_segments_rejected", False)
                    )
                except (OSError, KeyError, ValueError) as exc:
                    file_record["warnings"].append(
                        f"rebuilding unreadable/incomplete output: {exc}"
                    )
                    pending_features = dict(enabled_features)
                    append_mode = False
                else:
                    if all_segments_rejected:
                        file_record["status"] = "skipped_existing"
                        file_record["runtime_sec"] = round(time.time() - start, 3)
                        continue
                    pending_features = {
                        name: cfg
                        for name, cfg in enabled_features.items()
                        if name not in existing_groups
                    }
                    if not pending_features:
                        file_record["status"] = "skipped_existing"
                        file_record["runtime_sec"] = round(time.time() - start, 3)
                        continue
                    append_mode = True
            else:
                pending_features = dict(enabled_features)
                append_mode = False

            file_record["pending_features"] = sorted(pending_features.keys())
            registry = _build_registry(pending_features)

            input_meta = read_input_meta(input_path, use_labram_channels=use_labram_channels)
            if input_meta.eeg_indices.size == 0:
                raise ValueError("No EEG channels selected after channel policy.")

            run_bad_channel = bad_channel_enabled and not append_mode
            use_segment_parts = segment_part_names is not None
            n_segment_parts = len(segment_part_names) if segment_part_names else 1
            accumulator: Dict[str, Dict[str, List[np.ndarray]]] = {}
            hierarchical_records: List[
                Tuple[int, int, int, Dict[str, Dict[str, np.ndarray]]]
            ] = []
            warnings: List[str] = list(file_record["warnings"])
            bad_channel_masks: List[np.ndarray] = []
            bad_trial_mask: List[bool] = []
            bad_trial_reasons: List[str] = []
            unit_ids: List[str] = []
            unit_source_paths: List[str] = []
            segment_start_sec: List[float] = []
            parent_trial_index: List[int] = []
            n_trials = len(input_meta.trial_ids)
            bad_segments_by_trial: List[List[bool]] = [[] for _ in range(n_trials)]
            n_segments_seen = 0
            n_segments_rejected = 0
            segment_starts_by_trial: List[List[float]] = [[] for _ in range(n_trials)]
            segment_part_starts_by_trial: List[List[List[float]]] = [
                [] for _ in range(n_trials)
            ]
            segment_part_durations_by_trial: List[List[List[float]]] = [
                [] for _ in range(n_trials)
            ]

            existing_bad_mask = None
            n_existing_units = None
            if append_mode:
                existing_bad_mask = read_existing_bad_channel_mask(output_path)
                n_existing_units = read_existing_unit_count(output_path)
                if existing_bad_mask is not None and existing_bad_mask.shape[0] != n_existing_units:
                    warnings.append(
                        f"append: bad_channel_mask rows {existing_bad_mask.shape[0]} "
                        f"!= existing units {n_existing_units}; ignoring mask."
                    )
                    existing_bad_mask = None

            unit_cursor = 0
            for trial_index, signal, length in iter_trials(input_meta):
                trial_id = input_meta.trial_ids[trial_index]
                source_path = input_meta.source_paths[trial_index]
                trial_duration_sec = length / max(input_meta.sample_rate, 1e-6)

                trial_context = TrialContext(
                    sample_rate=input_meta.sample_rate,
                    channel_names=input_meta.eeg_channels,
                    trial_index=trial_index,
                    trial_id=trial_id,
                    source_path=source_path,
                    source_file=basename,
                )
                if run_bad_channel:
                    signal = bad_channel_preprocess(
                        signal,
                        trial_context,
                        enabled=True,
                        config=bad_channel_cfg,
                    )
                trial_mask = trial_context.bad_channel_mask
                trial_bad = trial_context.bad_trial
                trial_bad_reason = trial_context.bad_trial_reason
                warnings.extend(trial_context.warnings)

                if run_bad_channel and use_segment_parts:
                    bad_channel_masks.append(
                        np.asarray(trial_mask, dtype=bool)
                        if trial_mask is not None
                        else np.zeros(len(input_meta.eeg_channels), dtype=bool)
                    )
                    bad_trial_mask.append(bool(trial_bad))
                    bad_trial_reasons.append(trial_bad_reason)

                if segment_spec:
                    segments = list(
                        iter_segments(signal, input_meta.sample_rate, segment_spec)
                    )
                else:
                    segments = [(signal, 0.0, trial_duration_sec)]

                for seg_index, (seg_signal, start_sec, seg_duration_sec) in enumerate(segments):
                    n_segments_seen += 1
                    if use_segment_parts and existing_bad_mask is not None:
                        segment_channel_mask = existing_bad_mask[trial_index]
                    else:
                        segment_channel_mask = trial_mask

                    segment_rejected = False
                    reject_reason = ""
                    if trial_bad:
                        segment_rejected = True
                        reject_reason = (
                            f"parent_trial_bad"
                            + (f" ({trial_bad_reason})" if trial_bad_reason else "")
                        )
                    elif segment_reject_enabled:
                        segment_rejected = segment_exceeds_hard_peak(
                            seg_signal,
                            segment_channel_mask,
                            threshold_uv=hard_peak_threshold_uv,
                        )
                        if segment_rejected:
                            reject_reason = (
                                f"hard_peak_uv>{hard_peak_threshold_uv:g} "
                                "after bad-channel mask"
                            )
                    if segment_rejected:
                        n_segments_rejected += 1
                        warnings.append(
                            f"trial={trial_id} seg={seg_index:04d} rejected: {reject_reason}"
                        )

                    if use_segment_parts:
                        segment_starts_by_trial[trial_index].append(start_sec)
                        bad_segments_by_trial[trial_index].append(bool(segment_rejected))
                        if segment_rejected:
                            # Keep timeline slot; leave part times as empty placeholders.
                            segment_part_starts_by_trial[trial_index].append(
                                [float("nan")] * n_segment_parts
                            )
                            segment_part_durations_by_trial[trial_index].append(
                                [float("nan")] * n_segment_parts
                            )
                            continue

                        part_starts: List[float] = []
                        part_durations: List[float] = []
                        parts = list(
                            iter_equal_parts(
                                seg_signal,
                                input_meta.sample_rate,
                                n_segment_parts,
                            )
                        )
                        segment_cache: Dict[str, object] = {}
                        for part_index, (part_signal, part_offset_sec, part_duration_sec) in enumerate(parts):
                            part_name = segment_part_names[part_index]
                            part_start_sec = start_sec + part_offset_sec
                            unit_id = f"{trial_id}_seg{seg_index:04d}_{part_name}"
                            context = TrialContext(
                                sample_rate=input_meta.sample_rate,
                                channel_names=input_meta.eeg_channels,
                                trial_index=trial_index,
                                trial_id=trial_id,
                                source_path=source_path,
                                source_file=basename,
                                segment_index=seg_index,
                                segment_start_sec=start_sec,
                                segment_duration_sec=seg_duration_sec,
                                segment_part_index=part_index,
                                segment_part_name=part_name,
                                segment_part_start_sec=part_start_sec,
                                segment_part_duration_sec=part_duration_sec,
                                unit_id=unit_id,
                                bad_channel_mask=trial_mask,
                                bad_trial=trial_bad,
                                bad_trial_reason=trial_bad_reason,
                                cache=segment_cache,
                            )
                            if existing_bad_mask is not None:
                                context.bad_channel_mask = existing_bad_mask[trial_index]
                            if "source_network" in pending_features:
                                non_source_features = {
                                    name: cfg
                                    for name, cfg in pending_features.items()
                                    if name != "source_network"
                                }
                                outputs = registry.run(non_source_features, part_signal, context)
                                source_outputs = registry.run(
                                    {"source_network": pending_features["source_network"]},
                                    seg_signal,
                                    context,
                                )
                                outputs.update(source_outputs)
                            else:
                                outputs = registry.run(pending_features, part_signal, context)
                            hierarchical_records.append(
                                (trial_index, seg_index, part_index, outputs)
                            )
                            warnings.extend(context.warnings)
                            part_starts.append(part_start_sec)
                            part_durations.append(part_duration_sec)
                            unit_cursor += 1
                        segment_part_starts_by_trial[trial_index].append(part_starts)
                        segment_part_durations_by_trial[trial_index].append(part_durations)
                        continue

                    if segment_rejected:
                        continue

                    unit_id = (
                        f"{trial_id}_seg{seg_index:04d}"
                        if segment_spec
                        else trial_id
                    )
                    context = TrialContext(
                        sample_rate=input_meta.sample_rate,
                        channel_names=input_meta.eeg_channels,
                        trial_index=trial_index,
                        trial_id=trial_id,
                        source_path=source_path,
                        source_file=basename,
                        segment_index=seg_index,
                        segment_start_sec=start_sec,
                        segment_duration_sec=seg_duration_sec,
                        unit_id=unit_id,
                        bad_channel_mask=trial_mask,
                        bad_trial=trial_bad,
                        bad_trial_reason=trial_bad_reason,
                    )
                    if existing_bad_mask is not None:
                        context.bad_channel_mask = existing_bad_mask[unit_cursor]
                    outputs = registry.run(pending_features, seg_signal, context)
                    _aggregate(accumulator, outputs)
                    unit_ids.append(unit_id)
                    unit_source_paths.append(source_path)
                    segment_start_sec.append(start_sec)
                    parent_trial_index.append(trial_index)
                    if run_bad_channel:
                        bad_channel_masks.append(
                            np.asarray(trial_mask, dtype=bool)
                            if trial_mask is not None
                            else np.zeros(len(input_meta.eeg_channels), dtype=bool)
                        )
                        bad_trial_mask.append(bool(trial_bad))
                        bad_trial_reasons.append(trial_bad_reason)
                    warnings.extend(context.warnings)
                    unit_cursor += 1

            if append_mode and n_existing_units is not None:
                if use_segment_parts:
                    if n_trials != n_existing_units:
                        raise ValueError(
                            f"Input has {n_trials} trials but output H5 has "
                            f"{n_existing_units}; use matching segmentation config or --overwrite."
                        )
                elif len(unit_ids) != n_existing_units:
                    raise ValueError(
                        f"Segmentation produced {len(unit_ids)} units but output H5 has "
                        f"{n_existing_units}; use matching segmentation config or --overwrite."
                    )

            segment_valid_mask_array = None
            bad_segment_mask_array = None
            segment_part_valid_mask_array = None
            segment_part_start_array = None
            segment_part_duration_array = None
            output_trial_ids = unit_ids
            output_source_paths = unit_source_paths
            if use_segment_parts:
                max_segments = max((len(row) for row in segment_starts_by_trial), default=0)
                segment_start_array = np.full((n_trials, max_segments), np.nan, dtype=np.float32)
                segment_valid_mask_array = np.zeros((n_trials, max_segments), dtype=bool)
                bad_segment_mask_array = np.zeros((n_trials, max_segments), dtype=bool)
                segment_part_valid_mask_array = np.zeros(
                    (n_trials, max_segments, n_segment_parts), dtype=bool
                )
                segment_part_start_array = np.full(
                    (n_trials, max_segments, n_segment_parts), np.nan, dtype=np.float32
                )
                segment_part_duration_array = np.full(
                    (n_trials, max_segments, n_segment_parts), np.nan, dtype=np.float32
                )
                for trial_i, starts in enumerate(segment_starts_by_trial):
                    n_segments = len(starts)
                    if n_segments == 0:
                        continue
                    segment_start_array[trial_i, :n_segments] = np.asarray(
                        starts, dtype=np.float32
                    )
                    bad_flags = np.asarray(
                        bad_segments_by_trial[trial_i][:n_segments], dtype=bool
                    )
                    if bad_flags.shape[0] < n_segments:
                        bad_flags = np.pad(
                            bad_flags,
                            (0, n_segments - bad_flags.shape[0]),
                            constant_values=False,
                        )
                    bad_segment_mask_array[trial_i, :n_segments] = bad_flags
                    good_flags = ~bad_flags
                    segment_valid_mask_array[trial_i, :n_segments] = good_flags
                    segment_part_valid_mask_array[trial_i, :n_segments, :] = good_flags[
                        :, None
                    ]
                    part_starts_row = segment_part_starts_by_trial[trial_i]
                    part_durs_row = segment_part_durations_by_trial[trial_i]
                    if part_starts_row:
                        segment_part_start_array[trial_i, :n_segments, :] = np.asarray(
                            part_starts_row, dtype=np.float32
                        )
                    if part_durs_row:
                        segment_part_duration_array[trial_i, :n_segments, :] = np.asarray(
                            part_durs_row, dtype=np.float32
                        )
                feature_groups = _stack_hierarchical_records(
                    hierarchical_records,
                    n_trials=n_trials,
                    max_segments=max_segments,
                    n_parts=n_segment_parts,
                )
                _run_sanity_checks(feature_groups, warnings, segment_part_valid_mask_array)
                output_trial_ids = input_meta.trial_ids
                output_source_paths = input_meta.source_paths
            else:
                segment_start_array = (
                    np.asarray(segment_start_sec, dtype=np.float32) if segment_spec else None
                )
                feature_groups = _stack_accumulator(accumulator)
                _run_sanity_checks(feature_groups, warnings)
            bad_channel_mask_array = None
            bad_trial_mask_array = None
            bad_trial_reason_list: List[str] = []
            if run_bad_channel:
                bad_channel_mask_array = (
                    np.stack(bad_channel_masks, axis=0).astype(bool)
                    if bad_channel_masks
                    else np.zeros((n_trials, len(input_meta.eeg_channels)), dtype=bool)
                )
                bad_trial_mask_array = np.asarray(bad_trial_mask, dtype=bool)
                bad_trial_reason_list = [reason for reason in bad_trial_reasons if reason]

            bandpower_cfg = pending_features.get("bandpower", {})
            time_domain_cfg = pending_features.get("time_domain", {})
            attrs: Dict[str, Any] = {
                "pipeline_version": pipeline_version,
                "source_file": basename,
                "source_input_path": input_path,
                "sample_rate_hz": float(input_meta.sample_rate),
            }
            if segment_spec:
                attrs["segmentation_json"] = {
                    "duration_sec": segment_spec.duration_sec,
                    "overlap_sec": segment_spec.overlap_sec,
                    "min_duration_sec": segment_spec.min_duration_sec,
                    "thirds": (
                        {"enabled": True, "names": list(segment_part_names)}
                        if use_segment_parts
                        else {"enabled": False}
                    ),
                }
            if "bandpower" in pending_features:
                attrs["bands_json"] = bandpower_cfg.get("bands")
                attrs["welch_params_json"] = {
                    "window_sec": bandpower_cfg.get("welch_window_sec", 4.0),
                    "overlap": bandpower_cfg.get("welch_overlap", 0.5),
                    "analysis_range": bandpower_cfg.get("analysis_range", [0.5, 45.0]),
                }
            if not append_mode:
                attrs["bad_channel_json"] = bad_channel_cfg
                if bad_trial_mask_array is not None:
                    attrs["bad_trial_count"] = int(bad_trial_mask_array.sum())
                    attrs["bad_channel_mark_count"] = int(bad_channel_mask_array.sum())
                    attrs["bad_trial_reason_examples"] = bad_trial_reason_list[:20]
                if segment_reject_enabled:
                    attrs["segment_reject_json"] = {
                        "enabled": True,
                        "hard_peak_threshold_uv": hard_peak_threshold_uv,
                    }
                    attrs["bad_segment_count"] = int(n_segments_rejected)
                    attrs["segment_count"] = int(n_segments_seen)
                    attrs["all_segments_rejected"] = bool(
                        n_segments_seen > 0 and n_segments_rejected == n_segments_seen
                    )
            if time_domain_cfg.get("metrics") is not None:
                attrs["time_domain_metrics_json"] = time_domain_cfg.get("metrics")
            if "spatial" in pending_features:
                from eeg_captioner.features.features.spatial import (
                    DEFAULT_AP_RATIOS,
                    DEFAULT_BANDS,
                    DEFAULT_PAIRS,
                    DEFAULT_ROIS,
                )

                spatial_cfg = pending_features.get("spatial", {})
                spatial_rois = spatial_cfg.get("rois") or DEFAULT_ROIS
                spatial_bands = spatial_cfg.get("bands") or DEFAULT_BANDS
                spatial_pairs = spatial_cfg.get("pairs") or DEFAULT_PAIRS
                spatial_ap = spatial_cfg.get("ap_ratios") or DEFAULT_AP_RATIOS
                attrs["spatial_roi_names_json"] = list(spatial_rois.keys())
                attrs["spatial_band_names_json"] = list(spatial_bands.keys())
                attrs["spatial_pair_names_json"] = [
                    f"{p[0]}-{p[1]}" for p in spatial_pairs
                ]
                attrs["spatial_ap_ratio_names_json"] = list(spatial_ap.keys())
            if "source_network" in pending_features:
                source_cfg = pending_features.get("source_network", {})
                attrs["source_network_names_json"] = list(YEO7_NETWORK_NAMES)
                attrs["source_band_names_json"] = list(
                    (source_cfg.get("bands") or {}).keys()
                ) or ["delta", "theta", "alpha", "beta", "gamma"]
                attrs["source_assumptions_json"] = {
                    "montage": source_cfg.get("montage", "standard_1020"),
                    "inverse_method": source_cfg.get("inverse_method", "eLORETA"),
                    "noise_cov_method": source_cfg.get("noise_cov_method", "ad_hoc"),
                    "head_model": source_cfg.get("head_model", "bem"),
                    "source_spacing": source_cfg.get("source_spacing", "ico4"),
                    "average_reference": source_cfg.get("average_reference", True),
                    "artifacts_dir": source_cfg.get("artifacts_dir", "artifacts/source"),
                }

            atomic_new_write = not append_mode
            write_path = (
                f"{output_path}.tmp.{os.getpid()}.{shard_index}"
                if atomic_new_write
                else output_path
            )
            try:
                write_feature_h5(
                    output_path=write_path,
                    feature_groups=feature_groups,
                    channels=input_meta.eeg_channels,
                    trial_ids=output_trial_ids,
                    source_paths=output_source_paths,
                    attrs=attrs,
                    bad_channel_mask=bad_channel_mask_array,
                    bad_trial_mask=bad_trial_mask_array,
                    segment_start_sec=segment_start_array,
                    parent_trial_index=(
                        np.asarray(parent_trial_index, dtype=np.int32)
                        if segment_spec and not use_segment_parts
                        else None
                    ),
                    segment_valid_mask=segment_valid_mask_array,
                    bad_segment_mask=(
                        bad_segment_mask_array if use_segment_parts else None
                    ),
                    segment_part_names=(
                        segment_part_names if use_segment_parts else None
                    ),
                    segment_part_start_sec=segment_part_start_array,
                    segment_part_duration_sec=segment_part_duration_array,
                    append=append_mode,
                )
                if atomic_new_write:
                    os.replace(write_path, output_path)
            finally:
                if atomic_new_write and os.path.exists(write_path):
                    os.unlink(write_path)

            file_record["status"] = "appended" if append_mode else "ok"
            file_record["n_trials"] = n_trials
            file_record["n_units"] = (
                len(hierarchical_records) if use_segment_parts else len(unit_ids)
            )
            if use_segment_parts:
                file_record["n_segments_max"] = int(segment_valid_mask_array.shape[1])
                file_record["n_segments_valid"] = int(segment_valid_mask_array.sum())
                file_record["n_segments_bad"] = (
                    int(bad_segment_mask_array.sum())
                    if bad_segment_mask_array is not None
                    else 0
                )
                file_record["segment_parts"] = list(segment_part_names)
            file_record["n_segments_seen"] = int(n_segments_seen)
            file_record["n_segments_rejected"] = int(n_segments_rejected)
            file_record["n_channels_used"] = len(input_meta.eeg_channels)
            file_record["bad_trials"] = int(bad_trial_mask_array.sum()) if bad_trial_mask_array is not None else None
            file_record["bad_channel_marks"] = (
                int(bad_channel_mask_array.sum()) if bad_channel_mask_array is not None else None
            )
            file_record["warnings"] = sorted(set(warnings))
        except Exception as exc:  # noqa: BLE001
            file_record["status"] = "error"
            file_record["error"] = str(exc)
            file_record["traceback"] = traceback.format_exc()
        finally:
            file_record["runtime_sec"] = round(time.time() - start, 3)
            manifest["files"].append(file_record)
            status = str(file_record.get("status", "error"))
            if status in status_counts:
                status_counts[status] += 1
            else:
                status_counts["error"] += 1
            checkpoint = {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "num_shards": num_shards,
                "shard_index": shard_index,
                "processed": file_number,
                "files_shard_total": len(input_files),
                "file": basename,
                "status": status,
                "runtime_sec": file_record["runtime_sec"],
                "counts": dict(status_counts),
            }
            if status == "error":
                checkpoint["error"] = file_record.get("error", "unknown error")
            with open(progress_path, "a", encoding="utf-8") as progress_file:
                progress_file.write(json.dumps(checkpoint, ensure_ascii=False) + "\n")
            if (
                file_number % progress_every == 0
                or file_number == len(input_files)
                or status == "error"
            ):
                elapsed = time.monotonic() - run_started
                print(
                    json.dumps(
                        {
                            **checkpoint,
                            "files_per_min": file_number / max(elapsed, 1e-9) * 60.0,
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )

    ok_count = sum(1 for x in manifest["files"] if x["status"] in ("ok", "appended"))
    appended_count = sum(1 for x in manifest["files"] if x["status"] == "appended")
    skipped_count = sum(1 for x in manifest["files"] if x["status"] == "skipped_existing")
    error_count = sum(1 for x in manifest["files"] if x["status"] == "error")
    manifest["summary"] = {
        "ok": ok_count,
        "appended": appended_count,
        "skipped_existing": skipped_count,
        "error": error_count,
    }
    atomic_write_json(manifest_path, manifest)
    return manifest


def main() -> None:
    args = parse_args()
    config = apply_cli_overrides(load_config(args.config), args)
    manifest = run_pipeline(config)
    print(
        json.dumps(
            {
                "manifest_path": manifest["manifest_path"],
                "summary": manifest["summary"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if manifest["summary"]["error"]:
        raise RuntimeError(f"Feature extraction failed for {manifest['summary']['error']} files; see manifest")


if __name__ == "__main__":
    main()
