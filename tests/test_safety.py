from pathlib import Path
from types import SimpleNamespace
import pytest


def test_annotation_ignores_thirds_and_resumes(tmp_path, monkeypatch):
    pytest.importorskip("requests")
    from eeg_captioner.data.annotate import annotate
    from eeg_captioner.data import annotate as module
    summaries = tmp_path / "summaries"; summaries.mkdir()
    (summaries / "s__trial__seg0000.md").write_text("Synthetic complete-segment summary")
    (summaries / "s__trial__seg0000__early.md").write_text("Ignore diagnostic third")
    requests = []
    def post(*args, **kwargs):
        requests.append(kwargs)
        return SimpleNamespace(raise_for_status=lambda: None,
            json=lambda: {"choices": [{"message": {"content": "Synthetic EEG."}, "finish_reason": "stop"}]})
    monkeypatch.setenv("EEG_CAPTION_API_KEY", "test-placeholder-not-a-secret")
    monkeypatch.setattr(module.requests, "post", post)
    output = tmp_path / "captions"
    annotate(summaries, output, "https://example.invalid/v1", "test-model")
    annotate(summaries, output, "https://example.invalid/v1", "test-model")
    assert len(requests) == 1
    assert len(list(output.glob("*.txt"))) == 1
    assert "test-placeholder" not in (output / "annotation_config.json").read_text()


def test_duplicate_full_summary_fails(tmp_path):
    from eeg_captioner.data.summaries import segment_summaries
    for name in ("a", "b"):
        folder = tmp_path / name; folder.mkdir()
        (folder / "s__trial__seg0000.md").write_text("Synthetic")
    with pytest.raises(ValueError, match="Duplicate"):
        list(segment_summaries(tmp_path))


def test_missing_source_assets_fail_before_processing(tmp_path, monkeypatch):
    from eeg_captioner.features import source_artifacts
    from eeg_captioner.features.pipeline_runner import run_pipeline
    def unavailable(config):
        raise FileNotFoundError("Required source artifacts missing")
    monkeypatch.setattr(source_artifacts, "get_source_artifacts", unavailable)
    with pytest.raises(FileNotFoundError, match="artifacts"):
        run_pipeline({"input_root": str(tmp_path), "output_root": str(tmp_path / "out"),
                      "features": {"source_network": {"enabled": True}}})


def test_annotation_historical_settings_and_compaction(tmp_path, monkeypatch):
    pytest.importorskip("requests")
    from eeg_captioner.data.annotate import annotate
    from eeg_captioner.data import annotate as module
    from eeg_captioner.data.caption_input import compact_summary
    summaries = tmp_path / "summaries"; summaries.mkdir()
    text = "# **Segment-level EEG Numeric Summary**\n| Alpha | 0.123456 |\n"
    (summaries / "s__trial__seg0000.md").write_text(text)
    sent = []
    def post(*args, **kwargs):
        sent.append(kwargs["json"])
        return SimpleNamespace(raise_for_status=lambda: None,
            json=lambda: {"choices": [{"message": {"content": "Synthetic EEG."}, "finish_reason": "stop"}]})
    monkeypatch.setenv("EEG_CAPTION_API_KEY", "test-placeholder-not-a-secret")
    monkeypatch.setattr(module.requests, "post", post)
    annotate(summaries, tmp_path / "captions", "https://example.invalid/v1", "test-model",
             temperature=0.2, max_tokens=1200, compact_input=True)
    assert sent[0]["messages"][1]["content"] == compact_summary(text)
    assert "0.1235" in sent[0]["messages"][1]["content"]
    assert sent[0]["temperature"] == 0.2 and sent[0]["max_tokens"] == 1200
