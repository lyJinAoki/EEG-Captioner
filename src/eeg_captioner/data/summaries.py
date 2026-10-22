"""Select full 10-second summaries, excluding per-third diagnostic templates."""
from pathlib import Path
import re

SEGMENT_STEM = re.compile(r"(.+?)__(.+)__seg(\d+)")


def segment_summaries(root):
    seen = set()
    for path in sorted(Path(root).rglob("*__seg*.md")):
        if SEGMENT_STEM.fullmatch(path.stem) is None:
            continue
        if path.stem in seen:
            raise ValueError(f"Duplicate summary stem: {path.stem}")
        seen.add(path.stem)
        yield path
