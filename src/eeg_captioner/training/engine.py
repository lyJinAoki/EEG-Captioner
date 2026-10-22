"""Mainline training: classifier -> QA -> three joint adaptation stages."""
from contextlib import nullcontext
from pathlib import Path
import json
import math
import os
import random
import numpy as np
import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Sampler
import torch.distributed as dist
from eeg_captioner.io import read_jsonl, write_json, sha256
from eeg_captioner.data.dataset import PairDataset, load_manifest
from eeg_captioner.data.sampling import HybridHeadBatchSampler
from eeg_captioner.models.encoder import DualDomainSpatialEncoder
from eeg_captioner.models.heads import TaskSpecificQueryDecoders
from eeg_captioner.models.spectral import learning_rate_factor
from eeg_captioner.models.language_ops import forward_atomic_batch, scheduled_learning_rate
from eeg_captioner.evaluation.run import evaluate_probe, evaluate_language


class Probe(nn.Module):
    def __init__(self, questions, config):
        super().__init__()
        self.encoder = DualDomainSpatialEncoder(**{k: config[k] for k in ("d_model", "spatial_layers", "encoder_layers", "dropout")})
        self.decoders = TaskSpecificQueryDecoders(questions, d_model=config["d_model"],
                                                 decoder_layers=config.get("decoder_layers", 1), dropout=config["dropout"])

    def forward(self, eeg, heads):
        return self.decoders.forward_selected(self.encoder(eeg), heads)


class LanguageTrainingModel(nn.Module):
    def __init__(self, components):
        super().__init__()
        self.encoder, self.projector, self.qwen, self.tokenizer = components

    def train(self, mode=True):
        super().train(mode)
        self.encoder.eval()
        return self

    def forward(self, batch):
        device = next(self.projector.parameters()).device
        return forward_atomic_batch(batch, self.encoder, self.projector, self.qwen,
                                    self.qwen.get_input_embeddings(), self.tokenizer, device)[0]


class MixedSampler(Sampler):
    def __init__(self, labels, indices, questions, batch_size, batches, balanced_fraction, seed, fraction=0, rank=0, world=1):
        self.base = HybridHeadBatchSampler(labels, indices, questions, batch_size * world, batches, balanced_fraction, seed)
        self.indices, self.seed, self.fraction = np.asarray(indices), seed, fraction
        self.rank, self.world, self.batch_size = rank, world, batch_size
        self.start_batch = 0

    def __len__(self):
        return len(self.base) - self.start_batch

    def __iter__(self):
        rng = np.random.default_rng(self.seed + 71311)
        self.base.epoch = 0
        for j, qa in enumerate(self.base):
            if math.floor((j + 1) * self.fraction) > math.floor(j * self.fraction):
                count = self.batch_size * self.world
                qa = [(int(i), -1, -100) for i in rng.choice(self.indices, count, replace=count > len(self.indices))]
            if j < self.start_batch:
                continue
            start = self.rank * self.batch_size
            yield qa[start:start + self.batch_size]


def _unwrap(model):
    return model.module if isinstance(model, DDP) else model


def _save(path, model, optimizer, step, config, best_score, manifest_hash):
    path.mkdir(parents=True, exist_ok=True)
    model = _unwrap(model)
    if config["stage"] == "encoder":
        torch.save({"architecture": "dualdomain_spatial", "model": model.state_dict(),
                    "args": config, "step": step}, path / "model.pt")
    else:
        model.qwen.save_pretrained(path / "qwen_lora")
        model.tokenizer.save_pretrained(path / "qwen_lora")
        torch.save(model.projector.state_dict(), path / "eeg_projector.pt")
    torch.save({"step": step, "optimizer": optimizer.state_dict(), "config": config,
                "best_score": best_score, "manifest_sha256": manifest_hash,
                "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                "numpy_rng": np.random.get_state(), "python_rng": random.getstate()}, path / "training_state.pt")
    write_json(path / "config.json", config)


def train(config):
    config = dict(config)
    rank = int(os.environ.get("RANK", 0)); world = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    cuda = config.get("device", "cuda" if torch.cuda.is_available() else "cpu").startswith("cuda")
    device = torch.device(f"cuda:{local_rank}" if cuda else "cpu")
    if cuda:
        torch.cuda.set_device(device)
    if world > 1:
        dist.init_process_group("nccl" if cuda else "gloo")
    seed = int(config.get("seed", 42))
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if config["stage"] not in {"encoder", "qa", "joint"}:
        raise ValueError("stage must be encoder, qa or joint")
    output = Path(config["output_dir"])
    if output.exists() and any(output.iterdir()) and not config.get("resume"):
        raise FileExistsError(f"Use a new output_dir or explicit resume: {output}")
    if rank == 0:
        output.mkdir(parents=True, exist_ok=True)
    if world > 1:
        dist.barrier()
    rows = load_manifest(config["manifest"])
    digest = sha256(config["manifest"])
    questions = read_jsonl(Path(__file__).parents[1] / "data/questions_25.jsonl")
    indices = [i for i, row in enumerate(rows) if row["split"] == "train"]
    if not indices or not any(r["split"] == "val" for r in rows):
        raise ValueError("Training requires non-empty train and validation splits")
    encoder_stage = config["stage"] == "encoder"
    resume = Path(config["resume"]) if config.get("resume") else None
    if encoder_stage:
        model = Probe(questions, config).to(device)
        if resume:
            model.load_state_dict(torch.load(resume / "model.pt", map_location="cpu", weights_only=True)["model"], strict=True)
    else:
        from eeg_captioner.models.language import load_language
        initial = resume or config.get("initial_checkpoint")
        if config["stage"] == "joint" and initial is None:
            raise ValueError("Joint training requires the previous QA/joint checkpoint")
        components = load_language(config["qwen_path"], config["encoder_checkpoint"], initial,
                                   device, True, config["stage"], config.get("dtype", "bfloat16"))
        model = LanguageTrainingModel(components).to(device)
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=config["learning_rate"], weight_decay=config.get("weight_decay", 0.01))
    step, best_score = 0, (-1.0, -1.0)
    if resume:
        state = torch.load(resume / "training_state.pt", map_location="cpu", weights_only=False)
        if state["manifest_sha256"] != digest:
            raise ValueError("Resume manifest differs from the original training data")
        old = state["config"]
        for key in ("stage", "max_steps", "batch_size", "gradient_accumulation_steps", "world_size", "caption_fraction", "learning_rate"):
            if old.get(key, 1 if key == "world_size" else None) != (world if key == "world_size" else config.get(key)):
                raise ValueError(f"Resume configuration changed: {key}")
        optimizer.load_state_dict(state["optimizer"])
        step, best_score = int(state["step"]), tuple(state["best_score"])
        torch.set_rng_state(state["torch_rng"])
        if cuda and state["cuda_rng"]:
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        np.random.set_state(state["numpy_rng"]); random.setstate(state["python_rng"])
    config["world_size"] = world
    if world > 1:
        model = DDP(model, device_ids=[local_rank] if cuda else None, broadcast_buffers=False,
                    find_unused_parameters=encoder_stage)
    accumulation = int(config.get("gradient_accumulation_steps", 1))
    maximum = int(config["max_steps"])
    fraction = float(config.get("caption_fraction", 0)) if not encoder_stage else 0
    if not 0 <= fraction <= 1 or accumulation <= 0 or maximum <= 0:
        raise ValueError("Invalid training schedule")
    sampler = MixedSampler(np.asarray([r["labels"] for r in rows]), indices, questions, config["batch_size"],
                           maximum * accumulation, config.get("balanced_fraction", 0.5), seed, fraction, rank, world)
    sampler.start_batch = step * accumulation
    loader = DataLoader(PairDataset(rows, questions), batch_sampler=sampler,
                        num_workers=config.get("num_workers", 0), pin_memory=cuda)
    if rank == 0:
        write_json(output / "config.json", config)
        write_json(output / "data_provenance.json", {"manifest_sha256": digest, "evaluation": "independent_filename_split"})
    model.train(); optimizer.zero_grad(set_to_none=True)
    for batch_index, batch in enumerate(loader, start=step * accumulation):
        update = (batch_index + 1) % accumulation == 0
        sync = model.no_sync() if world > 1 and not update else nullcontext()
        with sync:
            amp = torch.autocast("cuda", dtype=torch.bfloat16) if cuda else nullcontext()
            with amp:
                if encoder_stage:
                    logits = model(batch["x"].to(device), batch["head"].to(device))
                    loss = nn.functional.cross_entropy(logits, batch["label"].to(device))
                else:
                    loss = model(batch)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at step {step}")
            (loss / accumulation).backward()
        if not update:
            continue
        torch.nn.utils.clip_grad_norm_(parameters, config.get("max_grad_norm", 1.0))
        if encoder_stage:
            lr = config["learning_rate"] * learning_rate_factor(step, config.get("warmup_steps", 1000), maximum, 0.1)
        else:
            lr = scheduled_learning_rate(step, maximum, 0, config["learning_rate"],
                                         config.get("cosine_decay", False), config.get("min_lr_ratio", 0.1))
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.step(); optimizer.zero_grad(set_to_none=True); step += 1
        if rank == 0 and (step % config.get("log_steps", 20) == 0 or step == 1):
            print(json.dumps({"step": step, "loss": float(loss.detach()), "lr": lr}), flush=True)
        if step % config.get("eval_steps", 500) == 0 or step == maximum:
            if world > 1:
                dist.barrier()
            if rank == 0:
                bare = _unwrap(model)
                if encoder_stage:
                    report = evaluate_probe(bare, rows, questions, "val", device, config.get("eval_per_class", 25), config.get("eval_batch_size", 16))
                else:
                    report, _, _ = evaluate_language((bare.encoder, bare.projector, bare.qwen, bare.tokenizer), rows, questions,
                                                     "val", device, config.get("eval_per_class", 25), config.get("caption_eval_cases", 24), config.get("eval_batch_size", 4),
                                                     config.get("qa_max_new_tokens", 12), config.get("caption_max_new_tokens", 320))
                score = (min(v["macro_head_balanced_accuracy"] for v in report["by_type"].values()), report["macro_head_balanced_accuracy"])
                report["step"] = step
                with (output / "history.jsonl").open("a") as f:
                    f.write(json.dumps(report) + "\n")
                if score > best_score:
                    best_score = score
                    _save(output / "best", model, optimizer, step, config, best_score, digest)
                _save(output / "last", model, optimizer, step, config, best_score, digest)
            if world > 1:
                dist.barrier()
            model.train()
        if step >= maximum:
            break
    if rank == 0:
        _save(output / "final", model, optimizer, step, config, best_score, digest)
        write_json(output / "COMPLETED.json", {"steps": step, "best_score": best_score})
    if world > 1:
        dist.barrier(); dist.destroy_process_group()
