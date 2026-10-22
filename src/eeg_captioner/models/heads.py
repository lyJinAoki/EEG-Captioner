import torch
from torch import nn
from torch.nn import functional as F
from eeg_captioner.data.rules import TYPE_OPTIONS
TASK_ORDER = tuple(TYPE_OPTIONS)
MAX_CLASSES = 5

class TaskSpecificQueryDecoders(nn.Module):
    """One independent decoder and five condition queries for each task."""

    def __init__(self, questions, d_model=256, decoder_layers=1, dropout=0.1):
        super().__init__()
        self.task_names = TASK_ORDER
        self.offsets = {}
        self.task_counts = {}
        offset = 0
        for task_name in self.task_names:
            count = sum(row["type"] == task_name for row in questions)
            if count != 5:
                raise RuntimeError(f"V5 expects five heads for {task_name}, got {count}")
            self.offsets[task_name] = offset
            self.task_counts[task_name] = count
            offset += count

        self.queries = nn.ParameterDict()
        self.decoders = nn.ModuleDict()
        self.classifiers = nn.ModuleDict()
        for task_name in self.task_names:
            query = nn.Parameter(torch.empty(5, d_model))
            nn.init.normal_(query, std=0.02)
            self.queries[task_name] = query
            layer = nn.TransformerDecoderLayer(
                d_model=d_model,
                nhead=8,
                dim_feedforward=d_model * 4,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.decoders[task_name] = nn.TransformerDecoder(
                layer, num_layers=decoder_layers
            )
            self.classifiers[task_name] = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, len(TYPE_OPTIONS[task_name])),
            )

    def forward_selected(self, memory, head_indices):
        output = memory.new_full((memory.shape[0], MAX_CLASSES), -1e4)
        for task_name in self.task_names:
            start = self.offsets[task_name]
            stop = start + 5
            mask = (head_indices >= start) & (head_indices < stop)
            if not mask.any():
                continue
            task_memory = memory[mask]
            queries = self.queries[task_name][None].expand(
                task_memory.shape[0], -1, -1
            )
            decoded = self.decoders[task_name](queries, task_memory)
            logits = self.classifiers[task_name](decoded)
            local_head = head_indices[mask] - start
            rows = torch.arange(task_memory.shape[0], device=memory.device)
            selected = logits[rows, local_head]
            # Under autocast, decoder logits are BF16 while memory (and thus
            # the preallocated padded output) can remain FP32.  IndexPut does
            # not perform implicit dtype promotion, so cast explicitly while
            # preserving the gradient path back into the decoder.
            output[mask, :selected.shape[-1]] = selected.to(output.dtype)
        return output

    def forward_all(self, memory):
        chunks = []
        for task_name in self.task_names:
            queries = self.queries[task_name][None].expand(memory.shape[0], -1, -1)
            decoded = self.decoders[task_name](queries, memory)
            logits = self.classifiers[task_name](decoded)
            if logits.shape[-1] < MAX_CLASSES:
                logits = F.pad(logits, (0, MAX_CLASSES - logits.shape[-1]), value=-1e4)
            chunks.append(logits)
        return torch.cat(chunks, dim=1)
