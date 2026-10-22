import json
from pathlib import Path
import numpy as np
import pytest
import torch
from scipy.signal import welch
from eeg_captioner.models.encoder import DualDomainSpatialEncoder
from eeg_captioner.models.spectral import FixedWelchPSD
from eeg_captioner.models.heads import TaskSpecificQueryDecoders
from eeg_captioner.models.projector import AtomicEEGProjector, weighted_causal_token_loss
from eeg_captioner.data.dataset import load_eeg, load_manifest
from eeg_captioner.data.rules import MODEL_CHANNELS, script_answer
from eeg_captioner.evaluation.caption_metrics import caption_overlap_metrics, corpus_bleu
from eeg_captioner.io import read_jsonl, write_jsonl
import eeg_captioner


@pytest.fixture
def questions():
    return read_jsonl(Path(eeg_captioner.__file__).parent / "data/questions_25.jsonl")


def test_welch_matches_scipy():
    rng = np.random.default_rng(71)
    raw = rng.normal(size=(2, 20, 2000)).astype(np.float32)
    model = FixedWelchPSD()
    actual = model(torch.from_numpy(raw.reshape(2, 20, 10, 200))).numpy()
    expected = []
    for start, stop in ((0, 666), (666, 1333), (1333, 2000)):
        freq, psd = welch(raw[..., start:stop], fs=200, nperseg=666, noverlap=333)
        expected.append(psd[..., (freq >= 0.5) & (freq <= 45)])
    np.testing.assert_allclose(actual, np.stack(expected, axis=2), atol=1e-7, rtol=5e-5)


def test_encoder_decoder_backward(questions):
    torch.manual_seed(7)
    model = DualDomainSpatialEncoder(d_model=16, spatial_layers=1, encoder_layers=1, dropout=0)
    heads = TaskSpecificQueryDecoders(questions, d_model=16, dropout=0)
    tokens = model(torch.randn(2, 20, 10, 200))
    assert tokens.shape == (2, 200, 16)
    logits = heads.forward_selected(tokens, torch.tensor([0, 20]))
    torch.nn.functional.cross_entropy(logits, torch.tensor([1, 2])).backward()
    assert model.time_projection[0].weight.grad.abs().sum() > 0
    assert model.frequency.projection[0].weight.grad.abs().sum() > 0


def test_scale_and_missing_oz(tmp_path):
    import h5py
    path = tmp_path / "record.h5"
    names = [n for n in MODEL_CHANNELS if n != "OZ"]
    with h5py.File(path, "w") as f:
        f.attrs["sample_rate"] = 200
        f["channels"] = np.asarray(names, dtype="S")
        f["trial_lengths_in_samples"] = [2000]
        f["data"] = np.ones((1, 19, 2000), dtype=np.float32) * 100
    x = load_eeg({"eeg_path": str(path), "trial_index": 0, "start_sample": 0})
    assert x.shape == (20, 10, 200)
    assert x[0].eq(1).all()
    assert x[10].eq(0).all()
    with pytest.raises(ValueError, match="length"):
        load_eeg({"eeg_path": str(path), "trial_index": 0, "start_sample": 1})


def test_split_leakage_rejected(tmp_path):
    path = tmp_path / "manifest.jsonl"
    base = dict(eeg_path="x.h5", summary_path="x.md", labels=[0] * 25)
    write_jsonl(path, [{**base, "id": "a", "split": "train"}, {**base, "id": "b", "split": "test"}])
    with pytest.raises(ValueError, match="multiple splits"):
        load_manifest(path)


def test_caption_metrics():
    text = "Alpha power increases in the early interval."
    assert caption_overlap_metrics(text, text)["rouge_l_f1"] == 1
    assert corpus_bleu([{"reference": text, "response": text}], 4) == 1
    assert corpus_bleu([{"reference": text, "response": ""}], 4) == 0


def test_loss_mask_matches_cross_entropy():
    torch.manual_seed(1)
    logits = torch.randn(2, 8, 12, requires_grad=True)
    labels = torch.randint(0, 12, (2, 8)); labels[:, :3] = -100
    loss, stats = weighted_causal_token_loss(logits, labels, labels.ne(-100))
    expected = torch.nn.functional.cross_entropy(logits[:, :-1].transpose(1, 2), labels[:, 1:], ignore_index=-100)
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert torch.isfinite(logits.grad).all()


def test_label_ambiguity(questions):
    q = questions[0]
    features = {"valid_channels": set(MODEL_CHANNELS), "bad_channels": set(),
                "absolute": {(q["period"], q["roi"]): dict(delta=10, theta=9.9, alpha=1, beta=1, gamma=1)},
                "relative": {}, "lateralization": {}}
    assert script_answer(q, features, 0.1)[0] is None

