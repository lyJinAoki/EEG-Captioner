import torch
from torch import nn
from torch.nn import functional as F

class AtomicEEGProjector(nn.Module):
    """Map one shared 200 x 256 EEG memory into Qwen's token space."""

    def __init__(self, llm_hidden_size: int):
        super().__init__()
        self.projector = nn.Sequential(
            nn.LayerNorm(256),
            nn.Linear(256, 1024),
            nn.GELU(),
            nn.Linear(1024, llm_hidden_size),
        )

    def forward(self, features):
        if features.ndim != 3 or tuple(features.shape[1:]) != (200, 256):
            raise ValueError(
                f"Expected shared EEG tokens [B,200,256], got {tuple(features.shape)}"
            )
        return self.projector(features)


def weighted_causal_token_loss(logits, labels, label_token_mask, label_weight=1.0):
    """Split causal-LM supervision into answer-label and formatting tokens.

    ``label_token_mask`` has the same unshifted shape as ``labels``.  The
    returned statistics are sums/counts so evaluation can aggregate them
    exactly across batches of different lengths.
    """

    if logits.ndim != 3 or labels.ndim != 2:
        raise ValueError("Expected logits [B,T,V] and labels [B,T]")
    if logits.shape[:2] != labels.shape or label_token_mask.shape != labels.shape:
        raise ValueError("Logits, labels, and label_token_mask must align")
    if label_weight <= 0:
        raise ValueError("label_weight must be positive")

    shift_logits = logits[:, :-1].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    valid = shift_labels.ne(-100)
    label_mask = label_token_mask[:, 1:].bool() & valid
    format_mask = valid & ~label_mask
    token_losses = F.cross_entropy(
        shift_logits.transpose(1, 2), shift_labels,
        ignore_index=-100, reduction="none",
    )
    weights = torch.where(
        label_mask,
        torch.as_tensor(label_weight, dtype=token_losses.dtype, device=logits.device),
        torch.ones((), dtype=token_losses.dtype, device=logits.device),
    )
    weighted_count = (weights * valid).sum().clamp_min(1)
    loss = (token_losses * weights * valid).sum() / weighted_count

    predictions = shift_logits.argmax(dim=-1)
    label_correct = predictions.eq(shift_labels) & label_mask
    label_present = label_mask.any(dim=1)
    label_exact = (label_correct | ~label_mask).all(dim=1) & label_present
    stats = {
        "full_loss_sum": (token_losses * valid).sum().detach(),
        "full_token_count": valid.sum().detach(),
        "label_loss_sum": (token_losses * label_mask).sum().detach(),
        "label_token_count": label_mask.sum().detach(),
        "format_loss_sum": (token_losses * format_mask).sum().detach(),
        "format_token_count": format_mask.sum().detach(),
        "label_token_correct": label_correct.sum().detach(),
        "label_exact_correct": label_exact.sum().detach(),
        "example_count": label_present.sum().detach(),
    }
    return loss, stats
