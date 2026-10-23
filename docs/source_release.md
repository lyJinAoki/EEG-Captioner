# Original feature-extraction release inventory

Reviewed on 2026-10-23 against the retained `eeg_feature_extraction` project.

| Original material | Public location | Decision |
| --- | --- | --- |
| `src/` feature plugins, QC, segmentation, H5 IO and source-artifact construction | `src/eeg_captioner/features/` | Already included; preserve the maintained package implementation rather than copying another standalone tree. |
| `scripts/fill_summary_templates.py` and summary templates | `src/eeg_captioner/features/summary.py`, `features/templates/` | Already included. |
| Main paper feature configuration | `configs/features.yaml` | Already included with portable paths. |
| `caption_gen.py` system prompt | `src/eeg_captioner/prompts/caption_annotation.txt` | Already included; the original provider routing and server defaults are unnecessary. |
| Historical generation parameters and `caption_unified.py:compact_summary` | `annotate` options and `data/caption_input.py` | Added explicit temperature/token/input-compaction options without copying credentials, quotas or routing machinery. |
| `outputs/caption/<batch>/*.txt` full-segment outputs | `annotations/captions/` | Preserve the retained collection as checked JSONL gzip shards. Distinguish it from the smaller historical training snapshot. |
| `outputs/summary_bin/bin_thresholds.json` | `src/eeg_captioner/data/bin_thresholds.json` | Already included as frozen calibration values. Re-fitting is a new calibration run, not recovery of the paper data. |
| Stress tests, cluster submission scripts, runtime caches and API routes | Not included | Environment-specific machinery is unnecessary for the portable pipeline. |
| Original EDF/H5 recordings, full feature outputs, source atlas/inverse caches | Not included | Obtain source data/assets separately and use the provided processing commands. |
| Pilot examples, raw API responses, internal notes and logs | Not included | Not needed by the public training interface; preserve only whitelisted provenance in the archive metadata. |

Exact source hashes and symbol mappings are recorded in [source_provenance.json](source_provenance.json). Archived captions are original text, while QA labels are derived during preparation. See [annotation documentation](../annotations/README.md) for hashes, use and snapshot limits.

## Historical snapshot search

Retained final-model configurations, dataset manifest, training scripts and experiment notes consistently point to a 160,781-sample matched cache and 25-head label cache. They record the seed and counts but do not contain the full matched sample list. The referenced project cache directory was absent during this review; a bounded search of code/work/scratch locations found no copy of those named caches. This is not proof that no backup exists elsewhere.

The project owner confirmed that caption generation continued after training data was cached. Therefore neither the later collection's sort order nor its size can substitute for the missing historical membership list. If that cache or an equivalent full manifest is recovered, export a separate versioned paper snapshot and validate its order, split membership and label alignment before claiming exact data reproduction.
