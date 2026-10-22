#!/usr/bin/env bash
set -euo pipefail
: "${QWEN_PATH:?Set QWEN_PATH to a local Qwen2.5-Omni-7B snapshot}"
eeg-captioner train --config configs/train_encoder.yaml
eeg-captioner export-encoder --source runs/encoder/best/model.pt --output checkpoints/encoder.pt
torchrun --standalone --nproc_per_node=2 -m eeg_captioner train --config configs/train_qa.yaml
torchrun --standalone --nproc_per_node=2 -m eeg_captioner train --config configs/train_joint_025.yaml
torchrun --standalone --nproc_per_node=2 -m eeg_captioner train --config configs/train_joint_075.yaml
torchrun --standalone --nproc_per_node=2 -m eeg_captioner train --config configs/train_joint_090.yaml
eeg-captioner export --encoder runs/encoder/best/model.pt --adapter runs/joint_090/final --output checkpoints/final
