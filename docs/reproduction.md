# Mainline reproduction

## Stages and weight identity

| Stage | Config | Input | Output selected by the chain |
|---|---|---|---|
| Dual-Domain classifier | train_encoder.yaml | Raw EEG and QA labels | encoder/best/model.pt |
| QA language adaptation | train_qa.yaml | Frozen classifier encoder | qa/best |
| Joint, caption fraction .25 | train_joint_025.yaml | QA best | joint_025/final |
| Joint, caption fraction .75 | train_joint_075.yaml | Previous final | joint_075/final |
| Joint, caption fraction .90 | train_joint_090.yaml | Previous final | joint_090/final |

The original final model is `relcap_cont2_15k_f090/final`, using the encoder
from `eeg_understanding_v51_dualdomain_spatial25/best.pt` at step 24,500.
Its historical report has QA Macro-BA 88.8111%, ROUGE-L 0.486486 and BLEU-4
0.330210. These are provenance values, not promises of a fresh run's scores.

Classifier pretraining uses the original five task-specific query decoders
(one TransformerDecoder layer and classification projection per task), rather
than replacing them with a different linear probe. The public encoder class
is copied symbol-for-symbol and keeps the original `state_dict` layout.

The language model is Qwen2.5-Omni-7B **Thinker**, with attention Q/K/V/O LoRA
rank 8, alpha 32 and dropout .05. Projector dimensions are 256 -> 1024 -> Qwen
hidden size, preceded by LayerNorm. Encoder outputs 200 tokens. Encoder and
base LLM weights remain frozen throughout language adaptation.

The three joint stages restart AdamW, preserve projector/LoRA weights, and use
LR 5e-5; the final stage decays to 5e-6. Microbatch/accumulation/world size are
1/2/2 (effective batch 4). The encoder stage uses batch 48 on one GPU.

## Evaluation differences that must remain explicit

Historical QA evaluation used 2,540 class-balanced questions sampled from the
union of val and test filenames. Both report names referred to that same
diagnostic set, which was also used to select checkpoints. This release's
default training selects only on val and reports held-out test separately.
Retraining under this cleaner protocol will not exactly reproduce the table.

QA Macro-BA is the mean of per-head balanced accuracies across the fixed
25 conditions (equivalently five equally weighted five-head family means).
Pooled per-family BA is also present in the report and is a different metric.
When a class or condition is absent in a small evaluation set, the report marks
missing support; such a small-set score is not comparable to the paper table.

Caption evaluation samples up to 500 held-out segments with the historical
seed rule and uses the original lowercase tokenization, LCS ROUGE-L F1 and
unsmoothed corpus BLEU-4. Teacher-forced caption loss is not exposed as a paper
metric by this release. Missing class/segment support must not be silently
interpreted as a fully balanced benchmark.

The original factual-consistency evaluation was external, web-based judging.
The required per-example judgments/model session provenance have not been found
in the supplied source artifacts. This release does not substitute heuristic
scores for GPT factual consistency.

## Restart and portability

New runs refuse a populated output directory. `resume` is for a trusted local
checkpoint from the same stage, with unchanged manifest and optimizer schedule.
It restores optimizer/RNG state and resumes after completed sampled batches.
Multi-GPU reduction order, library versions and hardware can affect results;
bitwise equivalence is not guaranteed. Mid-stage checkpoints are written at
evaluation boundaries. Only rank zero writes reports and checkpoints.

The default YAMLs contain no server paths, scheduler partition names or conda
environment assumptions. The shell chain uses standard `torchrun`. Adjust
`batch_size`, accumulation and world size together if GPU resources differ.

## Export

`export` copies only the encoder, projector, LoRA and tokenizer metadata. It
strips the local base-model path from adapter configuration and records SHA-256
hashes. Qwen base weights remain an external dependency. Optimizer state,
training caches, patient data, old experiment scripts and research histories
are excluded. Weights remain ignored by Git and can be distributed separately.

