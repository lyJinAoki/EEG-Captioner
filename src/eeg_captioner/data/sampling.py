import numpy as np
from torch.utils.data import Sampler
IGNORE_LABEL = -100
TASK_ORDER = ("dominant_band", "strongest_region", "peak_period", "temporal_pattern", "lateral_dominance")

class HybridHeadBatchSampler(Sampler):
    """Uniform heads with a mix of class-balanced and natural-class draws."""

    def __init__(
        self,
        labels,
        train_indices,
        questions,
        batch_size,
        batches_per_epoch,
        balanced_fraction,
        seed,
    ):
        if not 0.0 <= balanced_fraction <= 1.0:
            raise ValueError("balanced-fraction must be in [0, 1]")
        self.batch_size = int(batch_size)
        self.batches_per_epoch = int(batches_per_epoch)
        self.balanced_count = int(round(self.batch_size * balanced_fraction))
        self.seed = int(seed)
        self.epoch = 0
        train_indices = np.asarray(train_indices, dtype=np.int64)
        self.buckets = []
        self.natural = []
        self.natural_labels = []
        self.support = []
        for head_index, question in enumerate(questions):
            by_class = []
            class_support = []
            for class_index in range(len(question["options"])):
                members = train_indices[
                    labels[train_indices, head_index] == class_index
                ]
                if not len(members):
                    raise RuntimeError(
                        f"No train examples for head={head_index}, class={class_index}"
                    )
                by_class.append(members)
                class_support.append(int(len(members)))
            self.buckets.append(by_class)
            self.natural.append(np.concatenate(by_class))
            self.natural_labels.append(
                np.concatenate([
                    np.full(len(members), class_index, dtype=np.int64)
                    for class_index, members in enumerate(by_class)
                ])
            )
            self.support.append(class_support)

    def __len__(self):
        return self.batches_per_epoch

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        self.epoch += 1
        num_heads = len(self.buckets)
        head_stream = []
        class_orders = [
            rng.permutation(len(by_class)).tolist() for by_class in self.buckets
        ]
        class_offsets = [0] * num_heads
        for _ in range(self.batches_per_epoch):
            while len(head_stream) < self.batch_size:
                head_stream.extend(rng.permutation(num_heads).tolist())
            batch_heads = head_stream[:self.batch_size]
            del head_stream[:self.batch_size]
            balanced_slots = set(
                rng.choice(
                    self.batch_size,
                    size=self.balanced_count,
                    replace=False,
                ).tolist()
            )
            batch = []
            for slot, head_index in enumerate(batch_heads):
                if slot in balanced_slots:
                    order = class_orders[head_index]
                    offset = class_offsets[head_index]
                    if offset >= len(order):
                        order = rng.permutation(
                            len(self.buckets[head_index])
                        ).tolist()
                        class_orders[head_index] = order
                        offset = 0
                    class_index = int(order[offset])
                    class_offsets[head_index] = offset + 1
                    members = self.buckets[head_index][class_index]
                    sample_index = int(members[rng.integers(len(members))])
                else:
                    members = self.natural[head_index]
                    member_offset = int(rng.integers(len(members)))
                    sample_index = int(members[member_offset])
                    class_index = int(
                        self.natural_labels[head_index][member_offset]
                    )
                batch.append((sample_index, head_index, class_index))
            rng.shuffle(batch)
            yield batch
