import torch
import math
from .projector import weighted_causal_token_loss
from .prompts import prompt_parts


def unwrap(module):
    return module.module if hasattr(module, "module") else module


def warmup_learning_rate(step, warmup_steps, target_lr):
    return target_lr * min(1.0, (step + 1) / max(1, warmup_steps))

def tokenize_texts(tokenizer, texts, device, *, padding_side="right"):
    original_padding_side = tokenizer.padding_side
    try:
        tokenizer.padding_side = padding_side
        encoded = tokenizer(
            texts, padding=True, return_tensors="pt", add_special_tokens=False,
            truncation=False,
        )
    finally:
        tokenizer.padding_side = original_padding_side
    return encoded.input_ids.to(device), encoded.attention_mask.to(device)

def forward_atomic_batch(
    batch, frontend, projector, qwen, embedding, tokenizer, device,
    label_token_weight=1.0, eeg_ablation="normal",
):
    eeg = batch["x"].to(device, non_blocking=True)
    if eeg_ablation == "shuffle":
        if eeg.shape[0] < 2:
            raise ValueError("Shuffled EEG evaluation requires batch size >= 2")
        eeg = eeg.roll(1, dims=0)
    elif eeg_ablation == "zero":
        eeg = torch.zeros_like(eeg)
    elif eeg_ablation != "normal":
        raise ValueError(f"Unknown EEG ablation: {eeg_ablation}")
    features = frontend(eeg)
    eeg_tokens = projector(features)
    pieces = [prompt_parts(question, getattr(qwen.module if hasattr(qwen, "module") else qwen, "_eeg_system_prompt", None)) for question in batch["question"]]
    prefix_ids, prefix_mask = tokenize_texts(
        tokenizer, [item[0] for item in pieces], device
    )
    suffix_ids, suffix_mask = tokenize_texts(
        tokenizer, [item[1] for item in pieces], device, padding_side="left"
    )
    targets = [f"{label}<|im_end|>\n" for label in batch["target"]]
    target_ids, target_mask = tokenize_texts(tokenizer, targets, device)
    target_labels = target_ids.masked_fill(target_mask == 0, -100)
    label_ids, label_attention = tokenize_texts(
        tokenizer, list(batch["target"]), device
    )
    label_lengths = label_attention.sum(dim=1)
    target_positions = torch.arange(target_ids.shape[1], device=device)[None, :]
    target_label_mask = target_positions < label_lengths[:, None]
    eeg_tokens = eeg_tokens.to(embedding.weight.dtype)
    inputs_embeds = torch.cat((
        embedding(prefix_ids), eeg_tokens, embedding(suffix_ids),
        embedding(target_ids),
    ), dim=1)
    eeg_mask = torch.ones(eeg_tokens.shape[:2], dtype=torch.long, device=device)
    attention_mask = torch.cat(
        (prefix_mask, eeg_mask, suffix_mask, target_mask), dim=1
    )
    labels = torch.cat((
        torch.full_like(prefix_ids, -100),
        torch.full(eeg_tokens.shape[:2], -100, dtype=torch.long, device=device),
        torch.full_like(suffix_ids, -100),
        target_labels,
    ), dim=1)
    output = qwen(
        inputs_embeds=inputs_embeds, attention_mask=attention_mask,
        use_cache=False,
    )
    label_token_mask = torch.cat((
        torch.zeros_like(prefix_ids, dtype=torch.bool),
        torch.zeros(eeg_tokens.shape[:2], dtype=torch.bool, device=device),
        torch.zeros_like(suffix_ids, dtype=torch.bool),
        target_label_mask,
    ), dim=1)
    loss, metrics = weighted_causal_token_loss(
        output.logits, labels, label_token_mask, label_token_weight
    )
    return loss, metrics, labels, inputs_embeds.shape

@torch.inference_mode()
def generate_batch(
    batch, frontend, projector, qwen, tokenizer, device, max_new_tokens,
    eeg_ablation="normal",
):
    frontend = unwrap(frontend)
    qwen = unwrap(qwen)
    projector = unwrap(projector)
    pieces = [prompt_parts(question, getattr(qwen.module if hasattr(qwen, "module") else qwen, "_eeg_system_prompt", None)) for question in batch["question"]]
    prefix_ids, prefix_mask = tokenize_texts(
        tokenizer, [item[0] for item in pieces], device
    )
    suffix_ids, suffix_mask = tokenize_texts(
        tokenizer, [item[1] for item in pieces], device, padding_side="left"
    )
    eeg = batch["x"].to(device, non_blocking=True)
    if eeg_ablation == "shuffle":
        if eeg.shape[0] < 2:
            raise ValueError("Shuffled EEG evaluation requires batch size >= 2")
        eeg = eeg.roll(1, dims=0)
    elif eeg_ablation == "zero":
        eeg = torch.zeros_like(eeg)
    elif eeg_ablation != "normal":
        raise ValueError(f"Unknown EEG ablation: {eeg_ablation}")
    features = frontend(eeg)
    eeg_tokens = projector(features)
    embedding = qwen.get_input_embeddings()
    eeg_tokens = eeg_tokens.to(embedding.weight.dtype)
    inputs_embeds = torch.cat(
        (embedding(prefix_ids), eeg_tokens, embedding(suffix_ids)), dim=1
    )
    eeg_mask = torch.ones(eeg_tokens.shape[:2], dtype=torch.long, device=device)
    attention_mask = torch.cat((prefix_mask, eeg_mask, suffix_mask), dim=1)
    output = qwen.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        eos_token_id=tokenizer.convert_tokens_to_ids("<|im_end|>"),
        pad_token_id=tokenizer.pad_token_id,
    )
    return [text.strip() for text in tokenizer.batch_decode(output, skip_special_tokens=True)]

def scheduled_learning_rate(
    step, total_steps, warmup_steps, peak_lr, cosine_decay, min_lr_ratio,
):
    if step < 0:
        return 0.0
    if warmup_steps and step < warmup_steps:
        return warmup_learning_rate(step, warmup_steps, peak_lr)
    if not cosine_decay:
        return peak_lr
    decay_span = max(1, total_steps - warmup_steps - 1)
    progress = min(1.0, max(0.0, (step - warmup_steps) / decay_span))
    multiplier = min_lr_ratio + (1.0 - min_lr_ratio) * (
        0.5 * (1.0 + math.cos(math.pi * progress))
    )
    return peak_lr * multiplier
