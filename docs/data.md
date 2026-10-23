# Data and preprocessing

The TUEG-specific code is extracted from `build_tueg_h5.py`, not from the
multi-dataset inventory utility. It retains canonical channel handling,
0.1–75 Hz bandpass, 50 Hz notch, MNE resampling to 200 Hz and volts-to-microvolts
conversion. Recordings are grouped by subject and channel signature. Different
channel signatures produce separate H5 files. No new referencing is applied.

Preprocessing refuses to overwrite existing H5 files. Successful groups are
published from `.partial` files only after both passes succeed. Failures are
recorded in `preprocess_manifest.json` and cause a nonzero command exit. Keep
the original EDFs available when diagnosing a failed recording.

## Subject H5

- `data`: float32 `[trials, channels, padded_samples]`, microvolts.
- `channels`: canonical names; `trial_ids`, `source_paths`: recording identifiers.
- `trial_lengths_in_samples`: effective lengths, excluding padding.
- `sample_rate` attribute: 200.
- `labram_channel_indices`: historical standard-channel selection metadata;
  this does not imply a dependency on the LaBraM repository.

Feature extraction retains the original quality control: reference channels
marked bad, 500 µV peak/1 µV flat criteria, trial rejection thresholds, and
rejection of a 10-second segment if a remaining channel exceeds 100 µV.
It extracts 10-second non-overlapping segments and divides each into
`[0:666]`, `[666:1333]`, `[1333:2000]`. Feature plugins cover band powers,
time-domain statistics, spectral properties, spatial asymmetry and source networks.

The encoder's fixed Welch frontend uses 666-sample Hann windows at 200 Hz over
the same thirds, selecting 0.5–45 Hz bins. The model reads 20 channels in order:

```text
FP1 FP2 F3 F4 C3 C4 P3 P4 O1 O2 OZ F7 F8 T3 T4 T5 T6 FZ CZ PZ
```

Only missing OZ can be zero-filled, matching the historical loader. The model
receives `[batch,20,10,200]` scaled by `/100`; feature extraction uses µV directly.

Source-network artifacts are prepared explicitly with `source-artifacts` and
cached under the configured `artifacts_dir`. MNE may download fsaverage/Yeo
assets on first use. For an offline reduced smoke test, copy the YAML and set
`features.source_network.enabled: false`. This is not the paper annotation setup.

## Manifest

One JSON object per segment; paths are relative to the manifest location:

```json
{"id":"subject__trial__seg0000","eeg_path":"processed/subject.h5","summary_path":"summaries/0-1/template_v2_segments/subject__trial__seg0000.md","trial_index":0,"start_sample":0,"split":"train","caption":"...","labels":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]}
```

Labels follow the packaged 25-question ordering; `-100` means invalid and is
never sampled as a QA target. The preparation command defaults to the original
numeric validity plus bin-consistency filter, including quality-control channel
requirements. `--numeric-only` deliberately changes eligibility and is recorded
in manifest metadata. The shipped thresholds are frozen historical calibration
values; provenance should be considered before applying them to a new corpus.
The mainline does not need the older `train4` text-description training branch:
it applies its frozen bin thresholds directly to numerical features for QA
eligibility. Captions are generated from the full numerical segment summaries.

The annotation command selects full-segment stems ending in `__segNNNN` only;
per-third `__early`, `__middle`, and `__last` templates are diagnostic outputs,
not additional caption samples. Duplicate full-segment stems are rejected.

Caption text is embedded in the manifest, eliminating dependencies on external
caption files during training. EEG and summary paths are relocatable when moved
together with the manifest. Preparation records split counts, rejection counts,
question and threshold hashes. Duplicate sample IDs and cross-split file overlap
are rejected on loading.

Splitting uses the original seed-42 file-balancing algorithm. It is file-disjoint,
not guaranteed subject-disjoint: different H5 channel-signature groups of one
subject may land in different splits. Applications requiring subject-disjoint
evaluation should supply their own manifest with that grouping enforced.

The class-balanced training sampler requires at least one training example of
every class of every QA condition. A tiny subset can violate this requirement;
the trainer then reports the missing head/class rather than inventing targets.
Use the synthetic tests for a small complete-coverage smoke run.

## Retained caption archive

`prepare --captions annotations/captions` accepts the original checked JSONL gzip archive, as an alternative to directories of `.txt` files. See [annotation documentation](../annotations/README.md). The retained 204,460-caption collection is larger than the 160,781-sample historical training snapshot; exact snapshot membership is not yet recovered. Building a new manifest does not certify that it uses the paper's exact samples or partitions.
