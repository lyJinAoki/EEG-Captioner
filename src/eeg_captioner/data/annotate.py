"""Generate feature-grounded captions through an OpenAI-compatible HTTP endpoint."""
from pathlib import Path
import os
import time
import requests
from eeg_captioner.io import write_json
from .summaries import segment_summaries


def annotate(summaries, output, base_url, model, api_key_env="EEG_CAPTION_API_KEY", limit=None):
    key = os.environ.get(api_key_env)
    if not key:
        raise ValueError(f"Set {api_key_env}; credentials are never stored in the project")
    prompt = (Path(__file__).parents[1] / "prompts/caption_annotation.txt").read_text()
    paths = list(segment_summaries(summaries))
    if not paths:
        raise ValueError("No full-segment summaries found")
    if limit:
        paths = paths[:limit]
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    seen = set()
    for p in paths:
        if p.stem in seen:
            raise ValueError(f"Duplicate summary stem: {p.stem}")
        seen.add(p.stem)
        target = output / (p.stem + ".txt")
        if target.exists() and target.stat().st_size:
            continue
        for attempt in range(5):
            try:
                response = requests.post(base_url.rstrip("/") + "/chat/completions",
                    headers={"Authorization": f"Bearer {key}"}, timeout=180,
                    json={"model": model, "messages": [{"role": "system", "content": prompt},
                          {"role": "user", "content": p.read_text()}], "temperature": 0, "max_tokens": 2048})
                response.raise_for_status()
                choice = response.json()["choices"][0]
                text = choice["message"]["content"].strip()
                if not text or choice.get("finish_reason") == "length":
                    raise ValueError("Empty or truncated annotation")
                temporary = target.with_suffix(".tmp")
                temporary.write_text(text + "\n", encoding="utf-8"); temporary.replace(target)
                break
            except (requests.RequestException, ValueError, KeyError):
                if attempt == 4:
                    raise
                time.sleep(min(2 ** attempt, 16))
    write_json(output / "annotation_config.json", {"model": model, "base_url": base_url,
                                                   "temperature": 0, "prompt": prompt})
