#!/usr/bin/env python3
"""Package retained original captions without API responses or server paths."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from eeg_captioner.data.captions import SCHEMA, STEM, digest


def read_row(item):
    batch, path = item
    raw = path.read_bytes()
    text = raw.decode("utf-8")
    if not text.strip():
        raise ValueError(f"Empty caption: {path.name}")
    # Annotation text should contain physiological descriptions, not credentials
    # or local filesystem paths. Never print a matching value.
    if re.search(r"sk-[A-Za-z0-9_-]{20,}|/home/|/Users/|Bearer\s+\S+|[A-Za-z]:\\", text):
        raise ValueError(f"Unexpected private routing/path content in {path.name}")
    return {"id": path.stem, "caption": text, "source_batch": batch,
            "sha256": hashlib.sha256(raw).hexdigest()}


def package(source, output, shard_size=25000, workers=8):
    source, output = Path(source), Path(output)
    if shard_size < 1 or workers < 1:
        raise ValueError("shard_size and workers must be positive")
    if output.exists():
        raise FileExistsError(output)
    entries, seen, batches, usage = [], set(), {}, {}
    for batch in sorted(source.iterdir()):
        if not batch.is_dir():
            continue
        with os.scandir(batch) as es:
            names = sorted(e.name for e in es if e.name.endswith(".txt"))
        accepted = []
        for name in names:
            p = batch / name
            if STEM.fullmatch(p.stem) is None:
                continue
            if p.stem in seen:
                raise ValueError(f"Duplicate caption ID: {p.stem}")
            seen.add(p.stem)
            accepted.append((batch.name, p))
        if accepted:
            entries.extend(accepted)
            batches[batch.name] = len(accepted)
        usage_path = batch / "usage.json"
        if usage_path.is_file():
            d = json.loads(usage_path.read_text())
            # Last model is an API routing label, not a verified model revision.
            usage[batch.name] = {k: d[k] for k in ("last_model", "completed", "failed") if k in d}
    if not entries:
        raise ValueError("No original caption files found")
    output.mkdir(parents=True)
    meta = {"schema": SCHEMA, "records": len(entries), "source_batches": batches,
            "source": "eeg_feature_extraction/outputs/caption", "collection_date": datetime.now(timezone.utc).date().isoformat(),
            "paper_snapshot_records": 160781, "paper_snapshot_membership_recovered": False,
            "scope": "Retained caption collection after generation continued; not the recovered paper training snapshot.",
            "usage_summaries": usage, "shards": []}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for offset in range(0, len(entries), shard_size):
            shard = output / f"part-{offset // shard_size:05d}.jsonl.gz"
            rows = entries[offset:offset + shard_size]
            # Zero gzip timestamp and no filename make rebuilt shards deterministic.
            with shard.open("wb") as raw:
                with gzip.GzipFile(fileobj=raw, mode="wb", filename="", mtime=0) as gz:
                    for row in pool.map(read_row, rows):
                        gz.write((json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8"))
            meta["shards"].append({"file": shard.name, "records": len(rows), "bytes": shard.stat().st_size, "sha256": digest(shard)})
            print(f"Packaged {offset + len(rows)}/{len(entries)} captions", flush=True)
    (output / "manifest.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return meta


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--shard-size", type=int, default=25000)
    p.add_argument("--workers", type=int, default=8)
    a = p.parse_args()
    result = package(a.source, a.output, a.shard_size, a.workers)
    print(json.dumps({"records": result["records"], "shards": len(result["shards"])}))
