# Validation record

Validated locally on 2026-10-22. These are execution checks, not a new full-scale
reproduction of the paper's metrics.

## Automated tests

`pytest -q`: **15 passed**. All tests ran, including the optional MNE and
transformers/PEFT checks; none were skipped in the local validation environment.

- Fixed Welch features agree with SciPy; temporal/frequency branches and
  task-specific query decoders receive gradients.
- Channel order, missing-OZ handling, microvolt scaling, effective segment
  lengths and split-leakage checks.
- Actual MNE filtering/resampling and H5 writing on synthetic RawArray input.
  Only EDF file reading is mocked in this test.
- Synthetic H5 -> feature extraction -> segment summaries -> QA/caption
  manifest -> model input. Source-network computation is disabled in this test.
- Classifier optimization, save, encoder export, strict reload and resume.
- Real tiny Qwen Thinker forward/backward, PEFT adapter save/reload, and
  embedding-conditioned autoregressive generation (not a mocked LLM).
- QA training -> joint .25 -> joint .75 -> joint .90 -> held-out test evaluation,
  with real miniature Qwen weights, a synthetic tokenizer and synthetic EEG.
  This runs 1/4/4/10 optimizer steps and uses two-token evaluation for speed.
- Caption annotation with mocked HTTP: full-segment selection, resume,
  credential omission, and duplicate-summary rejection.
- Missing configured source artifacts fail explicitly rather than silently
  publishing a dataset without those features.

The short synthetic filtering fixture can emit an MNE filter-length warning;
it is intentionally shorter than a typical clinical recording. Torch's
norm-first nested-tensor warning does not prevent these checks.

## Real trained weights

The historical step-24,500 encoder and `relcap_cont2_15k_f090/final` adapter
were exported to a standalone bundle and strictly loaded. On an NVIDIA
A100-PCIE-40GB, Qwen2.5-Omni-7B Thinker + the exported adapter generated a
non-empty caption and QA answer from a real preprocessed EEG segment.
Encoder output was finite; peak allocated memory was approximately **16.9 GiB**.
This is a one-example compatibility check, not a benchmark accuracy estimate.

Loading Thinker from the complete Omni snapshot reports unused `talker` and
`token2wav` weights. These belong to speech-generation components not used by
EEG-Captioner; no missing Thinker weights were reported in this run.

Patient-derived predictions, local paths and validation logs are kept under
Git-ignored `local/`, and the 66 MiB model bundle under `checkpoints/final/`.
Neither is part of the source distribution.

## Distributed execution

A two-GPU NCCL/torchrun check completed two joint-training optimizer steps
with microbatch 1, accumulation 2, alternating QA/caption batches, frozen EEG
encoder, real tiny Qwen/PEFT, rank-zero validation and best/last/final saves.
This verifies the distributed training path on synthetic data, not convergence
or memory requirements of full 7B training.

## Environment and boundaries

Both wheel and source archive build successfully. The wheel was installed and
tested from outside the repository with the source-tree import path removed:
CLI help, packaged question/threshold/template/prompt assets, and strict loading
and forward execution of the real exported encoder all passed. The source
archive includes recipes, training script, tests and documentation, and excludes
local data, validation outputs and checkpoints. GitHub CI is supplied but has
not run remotely because this repository has not been published.

Python 3.10 on Linux aarch64, PyTorch 2.5.1+cu121, transformers 5.12.1,
PEFT 0.19.1, accelerate 1.15.0, MNE 1.12.1. Direct numeric-library versions are
recorded in `constraints-tested.txt`. Cluster-specific dynamic-library setup is
not included in the public recipes; install a working PyTorch stack for your
own machine.

Not rerun here: complete TUEG EDF conversion, source-space atlas downloading and
full-corpus inverse computation, paid caption annotation, full-length model
training, full test-set benchmark scoring, or external GPT factual judging.
The supplied configurations/code cover the first four workflows, but the smoke
checks must not be represented as executing them at corpus scale.

The historical QA reporting protocol differs from the independent val/test
protocol used by this release; see [reproduction](reproduction.md). No new paper
scores or fabricated factual-consistency results are claimed.

## Before public publication

- Review authorship, the proposed MIT source license and all upstream data/model
  terms with the project owners.
- Publish the portable adapter/encoder bundle separately (for example, a model
  hosting page or GitHub Release), then add its download URL and checksums.
  The Qwen base model is a separate download.
- Keep patient data, API credentials, cached models and private outputs out of
  Git. The source package excludes these directories and weight/data extensions.
- The local repository is initialized on `main`; no commit, remote creation or
  GitHub push is performed by this extraction.
