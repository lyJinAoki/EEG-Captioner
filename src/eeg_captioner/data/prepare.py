from pathlib import Path
import os
import re
import json
from collections import Counter
import h5py
from eeg_captioner.io import read_jsonl, write_jsonl, write_json, sha256
from .rules import parse_summary_features, script_answer, split_filenames, MODEL_CHANNELS
from .binning import Thresholds, audit_case
from .summaries import SEGMENT_STEM, segment_summaries


def prepare(processed, summaries, captions, output, bin_consistency=True, seed=42):
    processed, summaries, captions, output = map(Path, (processed, summaries, captions, output))
    questions_path = Path(__file__).with_name("questions_25.jsonl")
    thresholds_path = Path(__file__).with_name("bin_thresholds.json")
    questions = read_jsonl(questions_path)
    thresholds = Thresholds(thresholds_path)
    rows, rejected = [], Counter()
    cap_map = {}
    for p in sorted(captions.rglob("*.txt")):
        if p.stem in cap_map:
            raise ValueError(f"Duplicate caption stem: {p.stem}")
        cap_map[p.stem] = p
    metadata = {}
    for p in segment_summaries(summaries):
        match = SEGMENT_STEM.fullmatch(p.stem)
        filename, trial_id, segment = match.groups()
        raw = processed / (filename + ".h5")
        if p.stem not in cap_map:
            rejected["missing_caption"] += 1
            continue
        if raw not in metadata:
            with h5py.File(raw, "r") as f:
                decode = lambda v: v.decode() if isinstance(v, bytes) else str(v)
                metadata[raw] = ([decode(v) for v in f["trial_ids"][:]], {decode(v).upper() for v in f["channels"][:]})
        trials, channels = metadata[raw]
        if not (set(MODEL_CHANNELS) - {"OZ"}).issubset(channels):
            rejected["missing_channel"] += 1
            continue
        features = parse_summary_features(p)
        labels = []
        for h, q in enumerate(questions):
            label, reason = script_answer(q, features, 0.10)
            if label is not None and bin_consistency:
                case = {**q, "answer_label": label, "question_id": f"q{h + 1:04d}"}
                result = audit_case(case, features, thresholds)
                if not result["eligible"]:
                    label, reason = None, "bin/" + result["reason"]
            labels.append(-100 if label is None else q["options"].index(label))
            if reason:
                rejected[reason] += 1
        caption = cap_map[p.stem].read_text(encoding="utf-8").strip()
        if not caption:
            rejected["empty_caption"] += 1
            continue
        rows.append({"id": p.stem, "eeg_path": os.path.relpath(raw.resolve(), output.parent.resolve()),
                     "summary_path": os.path.relpath(p.resolve(), output.parent.resolve()),
                     "trial_index": trials.index(trial_id), "start_sample": int(segment) * 2000,
                     "caption": caption, "labels": labels})
    if not rows:
        raise ValueError("No matched EEG/summary/caption samples")
    files = split_filenames([(r["eeg_path"],) for r in rows], seed=seed)
    split_map = {filename: split for split, members in files.items() for filename in members}
    for r in rows:
        r["split"] = split_map[r["eeg_path"]]
    write_jsonl(output, rows)
    write_json(output.with_suffix(".metadata.json"), {
        "schema": 1, "segments": len(rows), "split_seed": seed,
        "splits": dict(Counter(r["split"] for r in rows)),
        "label_filter": "numeric_and_bin_consistency" if bin_consistency else "numeric_only",
        "questions_sha256": sha256(questions_path), "thresholds_sha256": sha256(thresholds_path),
        "rejections": dict(rejected), "eeg_units": "microvolts", "model_scale": 100.0,
        "note": "Filename-disjoint, not necessarily subject-disjoint when a subject has multiple H5 files."})
    return len(rows)
