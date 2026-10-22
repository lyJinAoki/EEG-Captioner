# EEG-Captioner

Grounding EEG-to-language generation in fundamental signal properties.

[![Model weights](https://img.shields.io/badge/Model-Hugging%20Face-yellow)](https://huggingface.co/LyJinAoki/EEG-Captioner)

This standalone implementation provides TUEG EDF preprocessing, deterministic
feature extraction, feature-grounded caption annotation, 25 signal-level QA
tasks, the Dual-Domain EEG Encoder, classifier pretraining, QA adaptation,
three-stage joint QA/caption training, inference, and evaluation.

The mainline does not import CBraMod, LaBraM, or any original research repository.
Baseline experiments and the full Table 2/Table 3 index are outside this release.

## Install

Use Python 3.10 or newer. Install a PyTorch build appropriate for your machine,
then install the package. The tested LLM API versions are constrained in
`pyproject.toml`; see [validation](docs/validation.md) for checks actually run.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[llm,preprocess,annotation,dev]'
eeg-captioner --help
pytest -q
```

Qwen2.5-Omni-7B must be downloaded separately. Point `QWEN_PATH` to its local
snapshot (containing config, tokenizer and weight shards). Model loading uses
local files only. A 7B model requires substantial RAM/VRAM; the paper's language
training recipes use two GPUs with microbatch 1 and gradient accumulation 2.
CPU tests use a small real Qwen model and synthetic signals.
`constraints-tested.txt` records the direct versions used in local validation;
pass `-c constraints-tested.txt` to pip when recreating that Python 3.10 stack.

```bash
export QWEN_PATH=/path/to/Qwen2.5-Omni-7B
```

## Use a trained model

A portable bundle contains `encoder.pt`, `eeg_projector.pt`, `qwen_lora/`, and
`bundle.json`. It does not contain the Qwen base model. Download the released
weights from [Hugging Face](https://huggingface.co/LyJinAoki/EEG-Captioner):

```bash
hf download LyJinAoki/EEG-Captioner --local-dir checkpoints/final
```

Existing trusted research weights can alternatively be exported:

```bash
eeg-captioner export \
  --encoder /path/to/eeg_understanding_v51_dualdomain_spatial25/best.pt \
  --adapter /path/to/relcap_cont2_15k_f090/final \
  --output checkpoints/final

eeg-captioner infer --bundle checkpoints/final --qwen-path "$QWEN_PATH" \
  --eeg-h5 data/processed/example.h5 --trial 0 --start-sample 0
```

Add `--question '...'` for signal-level QA. Inputs must be preprocessed H5
signals in **microvolts**, at **200 Hz**. The loader selects the fixed 20-channel
order, inserts zeros only for a missing OZ, and divides by 100 exactly once.
It rejects missing other channels, short segments and non-finite signals.

Only export trusted research `.pt` checkpoints: migration reads their Python
metadata. The portable encoder contains tensors and plain metadata and is
loaded using `weights_only=True` with strict state-dictionary matching.

## Build the dataset

Obtain TUEG v2.0.2 separately under its access conditions. No recordings or
derived patient samples are included in this source package.

```bash
# EDF_ROOT/partition/subject/session/montage/*.edf
eeg-captioner preprocess --edf-root /path/to/edf --output data/processed --workers 4

# Source-space features require MNE's template/atlas assets; prepare once.
eeg-captioner source-artifacts --config configs/features.yaml
eeg-captioner features --config configs/features.yaml
eeg-captioner summaries --input-glob 'data/features/*.h5' --output-dir data/summaries

# This command sends numerical summaries to your chosen annotation endpoint.
export EEG_CAPTION_API_KEY='your-key'
eeg-captioner annotate --summaries data/summaries --output data/captions \
  --base-url https://your-provider.example/v1 --model your-deepseek-model-id

eeg-captioner prepare --processed data/processed --summaries data/summaries \
  --captions data/captions --output data/manifest.jsonl
```

The annotation prompt is included in `src/eeg_captioner/prompts/`.
The paper used DeepSeek-V4-Flash; supply the corresponding model ID supported
by your provider. No provider credentials or original API routing are bundled.
Already generated captions can be placed in `data/captions` with names matching the
summary stem. Generation is not deterministic across provider/model updates.
Retain annotation metadata and original targets for reproducibility.

See [data format and preprocessing](docs/data.md), including optional omission
of source-network features for a reduced local test. The paper configuration
enables those features; omitting them changes caption annotation inputs.

## Train the complete mainline

Run from the repository root. Paths in the example YAML files are relative to
the current working directory; `${QWEN_PATH}` is expanded from the environment.

```bash
bash scripts/train_chain.sh
eeg-captioner evaluate --config configs/evaluate.yaml
```

The script runs classifier pretraining, QA adaptation, then 15k steps each at
caption fractions 0.25, 0.75 and 0.90. Each joint stage loads the previous stage's
projector and LoRA, uses a new optimizer, and keeps the same encoder frozen.
The last stage adds cosine LR decay. Each stage also works independently:

```bash
eeg-captioner train --config configs/train_encoder.yaml
torchrun --standalone --nproc_per_node=2 -m eeg_captioner train --config configs/train_qa.yaml
```

Configure a new `output_dir` for every run. To resume an interrupted run, add
`resume: runs/<run>/last` to its unchanged YAML. Resume restores optimizer and
RNG state and skips completed sampled microbatches. Cross-hardware/DDP resumption
is not claimed to be bit-for-bit identical.

## Reproduction protocol

This release preserves encoder/projector parameter names, input scaling,
question order, label rules, prompts and lexical metrics. It defaults to
**independent filename-disjoint validation and test evaluation**. Historical
paper QA scores used a combined val+test diagnostic set also used for model
selection; they must not be described as scores from this independent protocol.
See [reproduction notes](docs/reproduction.md).

GPT factual-consistency scores are not computed by the local lexical evaluator.
The original per-example web-based judgments are not included in the supplied
research artifacts. No replacement scores are fabricated.

## Layout

```text
configs/                       Feature, training and evaluation recipes
scripts/train_chain.sh         Complete mainline
src/eeg_captioner/
  preprocessing/               TUEG EDF -> subject H5
  features/                    Numeric features, source artifacts, summaries
  data/                        Manifest, QA rules, thresholds, sampling
  models/                      Encoder, query heads, projector, Qwen interface
  training/                    Classifier and distributed language training
  evaluation/                  QA BA, ROUGE-L and corpus BLEU-4
  prompts/                     Caption-annotation instructions
tests/                         Numerical, data and language smoke tests
docs/                          Data, reproduction, provenance and validation
```

MIT licensed source code; see `LICENSE` and `NOTICE`. Dataset, model weights and
template/atlas assets retain their separate upstream terms.
See the [publication checklist](docs/validation.md#before-public-publication)
before pushing the repository or publishing weights.
