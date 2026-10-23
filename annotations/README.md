# Original caption annotations

This directory contains **204,460 retained, pre-generated captions** from the original EEG feature-extraction project, in 9 gzip-compressed JSONL shards (41.9 MiB). The text is preserved byte-for-byte as UTF-8, including original whitespace. No annotation endpoint or API key is needed to use these captions.

## Use the original captions

From the repository root:

```bash
eeg-captioner verify-captions --captions annotations/captions

eeg-captioner prepare --processed data/processed --summaries data/summaries \
  --captions annotations/captions --output data/manifest.jsonl
```

The same `prepare` command also accepts an ordinary directory of `.txt` captions. It matches captions to locally available EEG files and full-segment numerical summaries by segment ID, computes QA labels, and creates the portable manifest. Raw EEG and numerical summaries must still be obtained/prepared separately. Matching this larger caption collection does **not** enforce the historical paper cohort.

## Collection versus paper snapshot

The paper experiments used **160,781 segments** with splits of 128,625 / 16,078 / 16,078. The project owner confirmed on 2026-10-23 that training used a partially generated caption collection and generation continued afterward. The present 204,460-caption collection is the retained later collection, **not a recovered list of exactly those 160,781 samples**.

Historical experiment configurations refer to `matched_eeg_summary_caption_cache_v3_20ch.pkl` and `understanding_probe_label_cache_v5_selected25.pkl`. Those caches were not present in the original project at release preparation. Retained configurations and the dataset manifest corroborate the counts and seed, but the exact training snapshot membership has not been recovered. Taking the first 160,781 records or re-splitting the entire collection is not an exact reconstruction.

The records deliberately have no invented train/validation/test assignment. Newly prepared manifests use the documented filename-disjoint split procedure; those splits must not be described as the verified historical partitions.

## Files and schema

`captions/manifest.json` lists total and per-batch counts, compressed-shard SHA-256 hashes, and a limited set of retained generation counters/model-routing labels. Each JSONL record contains:

- `id`: original full-segment stem used for EEG/summary matching.
- `caption`: unchanged UTF-8 caption text.
- `source_batch`: original generation batch.
- `sha256`: SHA-256 of the original caption file bytes.

The reader validates compressed-file hashes, per-caption hashes, IDs, duplicates, and record counts. Original API responses, credentials, provider routing, absolute server paths, recordings, EEG tensors, and unrelated pilot examples are excluded. Sample IDs preserve the de-identified source naming needed for pairing.

## Generation provenance

The paper names DeepSeek-V4-Flash. Retained usage metadata contains API labels such as `deepseek-chat`; an alias alone does not establish an immutable backend model revision. Usage counters are not necessarily equal to the number of retained files, since generation can resume or include previously existing outputs.

The original `caption_gen.py` uses temperature 0.2 and max_tokens 1200. The optional `caption_unified.py` compaction removes formatting and rounds numeric table entries to four significant digits. Its function is retained in the public code and can be selected when generating **new** annotations:

```bash
eeg-captioner annotate --summaries data/summaries --output data/new_captions \
  --base-url https://your-provider.example/v1 --model your-model-id \
  --temperature 0.2 --max-tokens 1200 --compact-input
```

Not every retained caption has a complete per-request model/settings record. These options preserve the available historical implementation; they do not promise identical regenerated captions. QA labels continue to use numerical summaries, rather than rounded annotation inputs.

## Rebuilding the archive

Maintainers with the original output tree can run:

```bash
python scripts/package_captions.py --source /path/to/original/outputs/caption \
  --output /path/to/new/archive --workers 8 --shard-size 25000
```

The packager refuses an existing destination, preserves original text, rejects duplicate IDs, excludes API response directories, and creates deterministic compressed shards for unchanged inputs. See [release selection](../docs/source_release.md) for the source-code inventory.

## Source data

This release contains generated caption text, not TUEG recordings or EEG tensors. Access to TUEG and upstream assets remains separate; the repository's source-code license does not grant access to those resources.
