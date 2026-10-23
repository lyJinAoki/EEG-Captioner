import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import pytest
from eeg_captioner.data.captions import digest, iter_captions, verify_captions


def packager():
    path = Path(__file__).parents[1] / "scripts/package_captions.py"
    spec = importlib.util.spec_from_file_location("caption_packager", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.package


def source_files(tmp_path):
    source = tmp_path / "original" / "batch-1"
    source.mkdir(parents=True)
    texts = {"s__t__seg0000": "Original caption.\n", "s__t__seg0001": "  Alpha activity.\r\n"}
    for key, text in texts.items():
        (source / (key + ".txt")).write_bytes(text.encode("utf-8"))
    (source / "usage.json").write_text(json.dumps({"last_model": "historical-alias", "api_key": "never-export-this"}))
    return source.parent, texts


def test_caption_archive_exact_roundtrip_and_provenance(tmp_path):
    source, texts = source_files(tmp_path)
    out = tmp_path / "archive"
    meta = packager()(source, out, shard_size=1, workers=1)
    assert dict(iter_captions(out)) == texts
    assert verify_captions(out) == {"records": 2, "integrity": "verified"}
    assert "api_key" not in json.dumps(meta)
    assert meta["paper_snapshot_membership_recovered"] is False
    assert dict(iter_captions(source)) == texts


def test_archive_detects_tampering_and_duplicate_ids(tmp_path):
    source, _ = source_files(tmp_path)
    out = tmp_path / "archive"
    packager()(source, out, workers=1)
    p = out / "part-00000.jsonl.gz"
    p.write_bytes(p.read_bytes() + b"bad")
    with pytest.raises(ValueError, match="checksum mismatch"):
        list(iter_captions(out))
    duplicate = source / "batch-2"
    duplicate.mkdir()
    (duplicate / "s__t__seg0000.txt").write_text("Duplicate")
    with pytest.raises(ValueError, match="Duplicate caption ID"):
        packager()(source, tmp_path / "other")


def test_archive_rejects_unsafe_shard_paths(tmp_path):
    source, _ = source_files(tmp_path)
    out = tmp_path / "archive"
    packager()(source, out, workers=1)
    p = out / "manifest.json"
    meta = json.loads(p.read_text())
    meta["shards"][0]["file"] = "../outside.jsonl.gz"
    p.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="Unsafe"):
        list(iter_captions(out))
