"""Manifest schema v1: paths are relative to the manifest, EEG is stored in µV."""
from pathlib import Path
import numpy as np
import h5py
import torch
from torch.utils.data import Dataset
from eeg_captioner.io import read_jsonl
from .rules import MODEL_CHANNELS


def load_manifest(path):
    path = Path(path).resolve()
    rows = read_jsonl(path)
    ids = set()
    split_by_file = {}
    for row in rows:
        if row["id"] in ids:
            raise ValueError(f"Duplicate sample ID: {row['id']}")
        ids.add(row["id"])
        for key in ("eeg_path", "summary_path"):
            row[key] = str((path.parent / row[key]).resolve())
        if len(row["labels"]) != 25:
            raise ValueError("Every sample needs 25 labels; invalid targets use -100")
        if row["split"] not in {"train", "val", "test"}:
            raise ValueError("Unknown split")
        key = row["eeg_path"]
        if key in split_by_file and split_by_file[key] != row["split"]:
            raise ValueError(f"EEG file appears in multiple splits: {key}")
        split_by_file[key] = row["split"]
    return rows


def load_eeg(row):
    with h5py.File(row["eeg_path"], "r") as f:
        if float(f.attrs.get("sample_rate", 0)) != 200:
            raise ValueError("Preprocessed EEG must be sampled at 200 Hz")
        names = [v.decode().upper() if isinstance(v, bytes) else str(v).upper() for v in f["channels"][:]]
        if len(names) != len(set(names)):
            raise ValueError("Duplicate channel names")
        start = int(row["start_sample"])
        trial = int(row["trial_index"])
        if trial < 0 or trial >= len(f["trial_lengths_in_samples"]):
            raise ValueError("Trial index out of range")
        if start < 0 or start + 2000 > int(f["trial_lengths_in_samples"][trial]):
            raise ValueError("Segment exceeds the effective trial length")
        window = np.asarray(f["data"][trial, :, start:start + 2000], dtype=np.float32)
    channels = []
    for name in MODEL_CHANNELS:
        if name in names:
            channels.append(window[names.index(name)])
        elif name == "OZ":
            channels.append(np.zeros(2000, dtype=np.float32))
        else:
            raise ValueError(f"Missing required channel: {name}")
    eeg = np.stack(channels)
    if not np.isfinite(eeg).all():
        raise ValueError("EEG contains NaN or infinity")
    return torch.from_numpy(eeg.reshape(20, 10, 200) / 100.0)


class PairDataset(Dataset):
    def __init__(self, rows, questions, fixed=None):
        self.rows, self.questions, self.fixed = rows, questions, fixed

    def __len__(self):
        return len(self.fixed) if self.fixed is not None else len(self.rows)

    def __getitem__(self, key):
        if self.fixed is not None:
            key = self.fixed[int(key)]
        index, head = int(key[0]), int(key[1])
        row = self.rows[index]
        if head == -1:
            from eeg_captioner.models.language import CAPTION_QUESTION
            if not row.get("caption", "").strip():
                raise ValueError(f"Missing caption: {row['id']}")
            question = CAPTION_QUESTION
            target = "<caption>\n" + row["caption"].strip() + "\n</caption>"
            label = -100
        else:
            label = int(row["labels"][head])
            q = self.questions[head]
            if label < 0 or label >= len(q["options"]):
                raise ValueError("Invalid QA target selected")
            question = q["question"]
            target = f"{chr(65 + label)}. {q['options'][label]}"
        return {"x": load_eeg(row), "sample_index": index, "head": head,
                "label": label, "question": question, "target": target, "id": row["id"]}


def eval_pairs(rows, questions, split, per_class=25, seed=42):
    """Stable within-head class balancing, strictly within the requested split."""
    rng = np.random.default_rng(seed)
    pairs = []
    for h, q in enumerate(questions):
        for label in range(len(q["options"])):
            candidates = [i for i, row in enumerate(rows) if row["split"] == split and row["labels"][h] == label]
            selected = candidates if per_class == 0 else rng.choice(candidates, min(per_class, len(candidates)), replace=False)
            pairs.extend((int(i), h) for i in selected)
    return pairs
