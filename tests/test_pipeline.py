from pathlib import Path
import json
import numpy as np
import h5py
import pytest
import torch
import yaml
from eeg_captioner.data.rules import MODEL_CHANNELS
from eeg_captioner.io import write_jsonl, read_jsonl


def make_h5(path, trials=1, sample_rate=200):
    rng = np.random.default_rng(72)
    signal = rng.normal(0, 5, (trials, 20, 2000)).astype(np.float32)
    signal += 12 * np.sin(2 * np.pi * 10 * np.arange(2000) / 200)[None, None, :]
    with h5py.File(path, "w") as f:
        f.attrs["sample_rate"] = sample_rate
        f["data"] = signal
        f["channels"] = np.asarray(MODEL_CHANNELS, dtype="S")
        f["trial_ids"] = np.asarray([f"trial{i}" for i in range(trials)], dtype="S")
        f["source_paths"] = np.asarray(["synthetic.edf"] * trials, dtype="S")
        f["trial_lengths_in_samples"] = [2000] * trials


def test_preprocess_filters_units_and_h5(tmp_path, monkeypatch):
    mne = pytest.importorskip("mne")
    from eeg_captioner.preprocessing import tueg
    names = ["EEG " + n + "-REF" for n in MODEL_CHANNELS]
    rng = np.random.default_rng(3)
    data = rng.normal(0, 5e-6, (20, 4000))
    raw = mne.io.RawArray(data, mne.create_info(names, 250, "eeg"), verbose=False)
    monkeypatch.setattr(tueg.mne.io, "read_raw_edf", lambda *a, **kw: raw.copy())
    result = tueg.process_group(["synthetic.edf"], str(tmp_path / "test.h5"), "s", list(MODEL_CHANNELS))
    assert result["status"] == "ok" and not result["errors"]
    with h5py.File(tmp_path / "test.h5", "r") as f:
        assert f.attrs["sample_rate"] == 200
        assert f["data"].shape == (1, 20, 3200)
        assert 0.1 < float(f["data"][:].std()) < 10


def test_features_summary_manifest(tmp_path):
    from eeg_captioner.features.pipeline_runner import run_pipeline
    from eeg_captioner.cli import main
    from eeg_captioner.data.prepare import prepare
    from eeg_captioner.data.dataset import load_manifest, load_eeg
    raw = tmp_path / "processed"; raw.mkdir()
    for i in range(3):
        make_h5(raw / f"s{i}.h5")
    config = yaml.safe_load((Path(__file__).parents[1] / "configs/features.yaml").read_text())
    config["input_root"] = str(raw); config["output_root"] = str(tmp_path / "features")
    config["features"]["source_network"]["enabled"] = False
    run_pipeline(config)
    main(["summaries", "--input-glob", str(tmp_path / "features/*.h5"), "--output-dir", str(tmp_path / "summaries")])
    from eeg_captioner.data.summaries import segment_summaries
    summaries = list(segment_summaries(tmp_path / "summaries"))
    assert len(summaries) == 3
    captions = tmp_path / "captions"; captions.mkdir()
    for p in summaries:
        (captions / (p.stem + ".txt")).write_text("Synthetic EEG caption for pipeline testing.")
    manifest = tmp_path / "manifest.jsonl"
    assert prepare(raw, tmp_path / "summaries", captions, manifest) == 3
    rows = load_manifest(manifest)
    assert {r["split"] for r in rows} == {"train", "val", "test"}
    assert load_eeg(rows[0]).shape == (20, 10, 200)


def test_encoder_training_checkpoint_and_resume(tmp_path):
    from eeg_captioner.training.engine import train
    from eeg_captioner.models.checkpoint import export_encoder, load_encoder
    import eeg_captioner
    questions = read_jsonl(Path(eeg_captioner.__file__).parent / "data/questions_25.jsonl")
    rows = []
    for split in ("train", "val", "test"):
        make_h5(tmp_path / (split + ".h5"), trials=5)
        for i in range(5):
            rows.append({"id": f"{split}{i}", "eeg_path": split + ".h5", "summary_path": "unused.md",
                         "trial_index": i, "start_sample": 0, "split": split, "caption": "Synthetic test caption.",
                         "labels": [i % len(q["options"]) for q in questions]})
    manifest = tmp_path / "manifest.jsonl"; write_jsonl(manifest, rows)
    config = {"stage": "encoder", "manifest": str(manifest), "output_dir": str(tmp_path / "run"),
              "seed": 42, "device": "cpu", "d_model": 16, "spatial_layers": 1, "encoder_layers": 1,
              "decoder_layers": 1, "dropout": 0, "batch_size": 2, "gradient_accumulation_steps": 1,
              "max_steps": 2, "learning_rate": 0.001, "warmup_steps": 0, "eval_steps": 1,
              "eval_per_class": 1, "eval_batch_size": 16}
    train(config)
    source = tmp_path / "run/final/model.pt"
    export_encoder(source, tmp_path / "encoder.pt")
    encoder = load_encoder(tmp_path / "encoder.pt")
    assert encoder(torch.randn(1, 20, 10, 200)).shape == (1, 200, 16)
    config["resume"] = str(tmp_path / "run/last")
    train(config)
    assert json.loads((tmp_path / "run/COMPLETED.json").read_text())["steps"] == 2
