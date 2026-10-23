"""Read original caption text files or checked, portable caption archives."""
from pathlib import Path
import gzip
import hashlib
import json
import re

SCHEMA = "eeg-captioner-captions-v1"
STEM = re.compile(r"[^/\\]+__[^/\\]+__seg\d+")


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def iter_captions(root):
    """Yield (segment ID, original text); validate all archive hashes/counts."""
    root = Path(root)
    manifest = root / "manifest.json"
    seen = set()
    if not manifest.is_file():
        for p in sorted(root.rglob("*.txt")):
            if STEM.fullmatch(p.stem) is None:
                continue
            if p.stem in seen:
                raise ValueError(f"Duplicate caption ID: {p.stem}")
            seen.add(p.stem)
            text = p.read_bytes().decode("utf-8")
            if not text.strip():
                raise ValueError(f"Empty caption: {p.stem}")
            yield p.stem, text
        return
    meta = json.loads(manifest.read_text(encoding="utf-8"))
    if meta.get("schema") != SCHEMA:
        raise ValueError("Unknown caption archive schema")
    count = 0
    for shard in meta["shards"]:
        name = shard["file"]
        if Path(name).name != name or "/" in name or "\\" in name or name in {".", ".."}:
            raise ValueError("Unsafe caption shard name")
        path = root / name
        if digest(path) != shard["sha256"]:
            raise ValueError(f"Caption shard checksum mismatch: {name}")
        shard_count = 0
        with gzip.open(path, "rt", encoding="utf-8", newline="") as f:
            for line in f:
                row = json.loads(line)
                key, text = row["id"], row["caption"]
                if STEM.fullmatch(key) is None or key in seen:
                    raise ValueError(f"Invalid or duplicate caption ID: {key}")
                if not isinstance(text, str) or not text.strip():
                    raise ValueError(f"Empty caption: {key}")
                if hashlib.sha256(text.encode("utf-8")).hexdigest() != row["sha256"]:
                    raise ValueError(f"Caption text checksum mismatch: {key}")
                seen.add(key)
                shard_count += 1
                count += 1
                yield key, text
        if shard_count != shard["records"]:
            raise ValueError(f"Caption shard count mismatch: {name}")
    if count != meta["records"]:
        raise ValueError("Caption archive total count mismatch")


def verify_captions(root):
    count = sum(1 for _ in iter_captions(root))
    if not count:
        raise ValueError("No captions found")
    return {"records": count, "integrity": "verified"}
