"""Build and cache MNE forward / inverse operators for source-network features."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np

_MNE_MODULE: Any | None = None
_MNE_IMPORT_ERROR: Exception | None = None

# Legacy TUH / LaBraM names -> MNE standard_1020 names.
CHANNEL_RENAME: Dict[str, str] = {
    "FP1": "Fp1",
    "FP2": "Fp2",
    "FZ": "Fz",
    "CZ": "Cz",
    "PZ": "Pz",
    "OZ": "Oz",
    "T3": "T7",
    "T4": "T8",
    "T5": "P7",
    "T6": "P8",
}

DEFAULT_DROP_CHANNELS: frozenset[str] = frozenset(
    {
        "A1",
        "A2",
        "T1",
        "T2",
        "EKG",
        "EKG1",
        "LOC",
        "LUC",
        "ROC",
        "RLC",
        "PG1",
        "PG2",
        "SP1",
        "SP2",
        "30",
    }
)

# Scalp channels used for source localization on sparse clinical montages.
DEFAULT_SOURCE_CHANNELS: Tuple[str, ...] = (
    "Fp1",
    "Fp2",
    "F3",
    "F4",
    "Fz",
    "C3",
    "C4",
    "Cz",
    "P3",
    "P4",
    "Pz",
    "O1",
    "O2",
    "T7",
    "T8",
    "P7",
    "P8",
)

YEO7_NETWORK_NAMES: Tuple[str, ...] = (
    "Vis",
    "SomMot",
    "DorsAttn",
    "SalVentAttn",
    "Limbic",
    "Cont",
    "Default",
)


@dataclass(frozen=True)
class SourceArtifacts:
    inv: Any
    labels: List[Any]
    src: Any
    subjects_dir: str
    channel_names: Tuple[str, ...]
    sample_rate_hz: float
    inverse_method: str
    noise_cov_method: str
    montage: str
    spacing: str
    head_model: str
    meta_path: str


def _load_mne() -> Any | None:
    global _MNE_MODULE, _MNE_IMPORT_ERROR
    if _MNE_MODULE is not None:
        return _MNE_MODULE
    if _MNE_IMPORT_ERROR is not None:
        return None
    try:
        import mne

        mne.set_log_level("ERROR")
        _MNE_MODULE = mne
        return mne
    except Exception as exc:  # noqa: BLE001
        _MNE_IMPORT_ERROR = exc
        return None


def mne_import_error() -> Exception | None:
    return _MNE_IMPORT_ERROR


def to_mne_channel_name(name: str) -> str:
    upper = str(name).strip().upper()
    if upper in CHANNEL_RENAME:
        return CHANNEL_RENAME[upper]
    return str(name).strip()


def select_source_channel_indices(
    channel_names: Sequence[str],
    *,
    drop_channels: frozenset[str] | None = None,
    allowed_channels: Sequence[str] | None = None,
    bad_channel_mask: np.ndarray | None = None,
) -> Tuple[List[int], List[str]]:
    """Map trial channels to the source-localization montage."""
    drop = drop_channels or DEFAULT_DROP_CHANNELS
    allowed = {c.strip() for c in (allowed_channels or DEFAULT_SOURCE_CHANNELS)}
    allowed_upper = {c.upper() for c in allowed}

    indices: List[int] = []
    mne_names: List[str] = []
    for idx, raw_name in enumerate(channel_names):
        if bad_channel_mask is not None and bool(bad_channel_mask[idx]):
            continue
        upper = str(raw_name).strip().upper()
        if upper in drop:
            continue
        mne_name = to_mne_channel_name(raw_name)
        if mne_name.upper() not in allowed_upper:
            continue
        indices.append(idx)
        mne_names.append(mne_name)
    return indices, mne_names


def _meta_path(artifacts_dir: Path) -> Path:
    return artifacts_dir / "source_artifacts_meta.json"


def _write_meta(path: Path, payload: Mapping[str, object]) -> None:
    path.write_text(json.dumps(dict(payload), indent=2, ensure_ascii=False), encoding="utf-8")


def _collapse_yeo7_bilateral_labels(labels: List[Any], mne: Any) -> List[Any]:
    """Merge lh/rh Yeo-7 labels into seven bilateral networks."""
    grouped: Dict[str, List[Any]] = {}
    for label in labels:
        if "Medial_Wall" in label.name:
            continue
        net_key = label.name.rsplit("-", 1)[0]
        grouped.setdefault(net_key, []).append(label)

    merged: List[Any] = []
    for net_id in range(1, 8):
        key = f"7Networks_{net_id}"
        parts = grouped.get(key, [])
        if len(parts) != 2:
            raise RuntimeError(f"Expected bilateral labels for {key}, got {len(parts)}")
        merged.append(parts[0] + parts[1])
    if len(merged) != len(YEO7_NETWORK_NAMES):
        raise RuntimeError(f"Expected {len(YEO7_NETWORK_NAMES)} merged Yeo-7 labels, got {len(merged)}")
    return merged


def _load_yeo7_labels(mne: Any, subject: str, subjects_dir: str) -> List[Any]:
    raw_labels = mne.read_labels_from_annot(
        subject,
        parc="Yeo2011_7Networks_N1000",
        subjects_dir=subjects_dir,
        verbose="ERROR",
    )
    return _collapse_yeo7_bilateral_labels(raw_labels, mne)


def build_source_artifacts(
    config: Mapping[str, object],
    *,
    force_rebuild: bool = False,
) -> SourceArtifacts:
    """Build or load cached forward/inverse operators and Yeo-7 labels."""
    mne = _load_mne()
    if mne is None:
        raise RuntimeError(f"mne is unavailable: {_MNE_IMPORT_ERROR!s}")

    artifacts_dir = Path(str(config.get("artifacts_dir", "artifacts/source")))
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    sample_rate_hz = float(config.get("sample_rate_hz", 200.0))
    channel_names = tuple(str(c) for c in config.get("channels", DEFAULT_SOURCE_CHANNELS))
    inverse_method = str(config.get("inverse_method", "eLORETA"))
    noise_cov_method = str(config.get("noise_cov_method", "ad_hoc"))
    montage = str(config.get("montage", "standard_1020"))
    spacing = str(config.get("source_spacing", "ico4"))
    head_model = str(config.get("head_model", "sphere")).lower()
    subject = str(config.get("subject", "fsaverage"))
    loose = float(config.get("loose", 0.2))
    depth = float(config.get("depth", 0.8))

    inv_path = artifacts_dir / "inv.fif"
    src_path = artifacts_dir / "src-src.fif"
    meta_file = _meta_path(artifacts_dir)

    expected_meta = {
        "sample_rate_hz": sample_rate_hz,
        "channel_names": list(channel_names),
        "inverse_method": inverse_method,
        "noise_cov_method": noise_cov_method,
        "montage": montage,
        "spacing": spacing,
        "head_model": head_model,
        "subject": subject,
        "loose": loose,
        "depth": depth,
    }

    if (
        not force_rebuild
        and inv_path.exists()
        and src_path.exists()
        and meta_file.exists()
    ):
        stored = json.loads(meta_file.read_text(encoding="utf-8"))
        comparable_stored = {k: stored.get(k) for k in expected_meta.keys()}
        if comparable_stored == expected_meta:
            inv = mne.minimum_norm.read_inverse_operator(str(inv_path), verbose="ERROR")
            src = mne.read_source_spaces(str(src_path), verbose="ERROR")
            subjects_dir = str(stored.get("subjects_dir", ""))
            labels = _load_yeo7_labels(mne, subject, subjects_dir)
            return SourceArtifacts(
                inv=inv,
                labels=labels,
                src=src,
                subjects_dir=subjects_dir,
                channel_names=channel_names,
                sample_rate_hz=sample_rate_hz,
                inverse_method=inverse_method,
                noise_cov_method=noise_cov_method,
                montage=montage,
                spacing=spacing,
                head_model=head_model,
                meta_path=str(meta_file),
            )

    fs_dir = mne.datasets.fetch_fsaverage(verbose=False)
    subjects_dir = os.path.dirname(fs_dir)

    info = mne.create_info(list(channel_names), sfreq=sample_rate_hz, ch_types="eeg")
    info.set_montage(montage, on_missing="raise", verbose="ERROR")

    src = mne.setup_source_space(
        subject,
        spacing=spacing,
        subjects_dir=subjects_dir,
        add_dist=False,
        verbose="ERROR",
    )
    mne.write_source_spaces(str(src_path), src, overwrite=True, verbose="ERROR")

    mindist = float(config.get("forward_mindist", 5.0))
    if head_model == "sphere":
        bem = mne.make_sphere_model("auto", info=info, verbose="ERROR")
        mindist = float(config.get("forward_mindist", 0.0))
    elif head_model == "bem":
        bem_sol = os.path.join(subjects_dir, subject, "bem", f"{subject}-5120-5120-5120-bem-sol.fif")
        if not os.path.exists(bem_sol):
            bem_sol = os.path.join(subjects_dir, subject, "bem", f"{subject}-5120-bem-sol.fif")
        if not os.path.exists(bem_sol):
            raise FileNotFoundError(
                f"BEM solution not found under {subjects_dir}/{subject}/bem; "
                "run mne.datasets.fetch_fsaverage() first."
            )
        bem = mne.read_bem_solution(bem_sol, verbose="ERROR")
    else:
        raise ValueError(f"Unsupported head_model: {head_model} (use sphere or bem)")

    fwd = mne.make_forward_solution(
        info,
        trans="fsaverage",
        src=src,
        bem=bem,
        eeg=True,
        meg=False,
        mindist=mindist,
        verbose="ERROR",
    )
    mne.write_forward_solution(str(artifacts_dir / "fwd.fif"), fwd, overwrite=True, verbose="ERROR")
    if not np.isfinite(fwd["sol"]["data"]).all():
        raise RuntimeError(
            "Forward solution contains non-finite values; try head_model=bem or lower forward_mindist."
        )

    if noise_cov_method == "ad_hoc":
        noise_cov = mne.make_ad_hoc_cov(info, dict(eeg=1e-12), verbose="ERROR")
    else:
        raise ValueError(f"Unsupported noise_cov_method: {noise_cov_method}")

    inv = mne.minimum_norm.make_inverse_operator(
        info,
        fwd,
        noise_cov,
        loose=loose,
        depth=depth,
        verbose="ERROR",
    )
    mne.minimum_norm.write_inverse_operator(str(inv_path), inv, overwrite=True, verbose="ERROR")

    labels = _load_yeo7_labels(mne, subject, subjects_dir)

    expected_meta["subjects_dir"] = subjects_dir
    _write_meta(meta_file, expected_meta)

    return SourceArtifacts(
        inv=inv,
        labels=labels,
        src=src,
        subjects_dir=subjects_dir,
        channel_names=channel_names,
        sample_rate_hz=sample_rate_hz,
        inverse_method=inverse_method,
        noise_cov_method=noise_cov_method,
        montage=montage,
        spacing=spacing,
        head_model=head_model,
        meta_path=str(meta_file),
    )


_ARTIFACT_CACHE: Dict[str, SourceArtifacts] = {}


def get_source_artifacts(config: Mapping[str, object]) -> SourceArtifacts:
    """Return cached artifacts for a config fingerprint."""
    artifacts_dir = os.path.abspath(str(config.get("artifacts_dir", "artifacts/source")))
    key = artifacts_dir
    if key not in _ARTIFACT_CACHE:
        cfg = dict(config)
        cfg["artifacts_dir"] = artifacts_dir
        _ARTIFACT_CACHE[key] = build_source_artifacts(cfg)
    return _ARTIFACT_CACHE[key]
