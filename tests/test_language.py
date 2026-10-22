"""Small real Qwen Thinker exercises embedding injection, PEFT and generation on CPU."""
import pytest
import torch
from eeg_captioner.models.language_ops import forward_atomic_batch, generate_batch
from eeg_captioner.models.projector import AtomicEEGProjector


class Tokenizer:
    padding_side = "right"
    pad_token_id = 0

    def __call__(self, texts, **kwargs):
        from types import SimpleNamespace
        encoded = [[3 + ord(c) % 24 for c in text] for text in texts]
        n = max(map(len, encoded))
        ids, masks = [], []
        for row in encoded:
            padding = [0] * (n - len(row))
            ids.append(padding + row if self.padding_side == "left" else row + padding)
            masks.append([int(v != 0) for v in ids[-1]])
        return SimpleNamespace(input_ids=torch.tensor(ids), attention_mask=torch.tensor(masks))

    def convert_tokens_to_ids(self, text):
        return 2

    def batch_decode(self, values, **kwargs):
        return [" ".join(map(str, row.tolist())) for row in values]


def tiny_config():
    pytest.importorskip("transformers")
    from transformers import Qwen2_5OmniThinkerConfig
    config = Qwen2_5OmniThinkerConfig(text_config={"vocab_size": 32, "hidden_size": 16,
        "intermediate_size": 32, "num_hidden_layers": 1, "num_attention_heads": 2,
        "num_key_value_heads": 2, "max_position_embeddings": 4096,
        "rope_parameters": {"rope_type": "default", "mrope_section": [1, 1, 2], "rope_theta": 1000000.0}},
        audio_config={"num_mel_bins": 8, "encoder_layers": 1, "encoder_attention_heads": 2,
                      "encoder_ffn_dim": 32, "d_model": 16, "output_dim": 16},
        vision_config={"depth": 1, "hidden_size": 16, "intermediate_size": 32,
                       "num_heads": 2, "out_hidden_size": 16, "fullatt_block_indexes": [0]})
    config.vision_start_token_id = 29
    return config


def test_real_tiny_qwen_training_generation_adapter(tmp_path):
    pytest.importorskip("transformers")
    peft = pytest.importorskip("peft")
    from transformers import Qwen2_5OmniThinkerForConditionalGeneration
    from eeg_captioner.models.language import LORA_TARGETS
    config = tiny_config()
    base = Qwen2_5OmniThinkerForConditionalGeneration(config)
    model = peft.get_peft_model(base, peft.LoraConfig(task_type="CAUSAL_LM", r=8, lora_alpha=32, target_modules=LORA_TARGETS))
    projector = AtomicEEGProjector(16)
    frontend = torch.nn.Identity()
    tokenizer = Tokenizer()
    batch = {"x": torch.randn(2, 200, 256), "question": ["Band?", "What band?"], "target": ["A. delta", "B. theta"]}
    loss, *_ = forward_atomic_batch(batch, frontend, projector, model, model.get_input_embeddings(), tokenizer, "cpu")
    loss.backward()
    assert torch.isfinite(loss)
    assert projector.projector[1].weight.grad.abs().sum() > 0
    model.save_pretrained(tmp_path / "adapter")
    restored = peft.PeftModel.from_pretrained(Qwen2_5OmniThinkerForConditionalGeneration(config), tmp_path / "adapter")
    result = generate_batch(batch, frontend, projector.eval(), restored.eval(), tokenizer, "cpu", 2)
    assert len(result) == 2


def test_language_training_chain_and_evaluation(tmp_path):
    pytest.importorskip("transformers")
    pytest.importorskip("peft")
    from transformers import Qwen2_5OmniThinkerForConditionalGeneration, PreTrainedTokenizerFast
    from tokenizers import Tokenizer as BackendTokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from pathlib import Path
    import json
    import eeg_captioner
    from test_pipeline import make_h5
    from eeg_captioner.io import read_jsonl, write_jsonl
    from eeg_captioner.models.encoder import DualDomainSpatialEncoder
    from eeg_captioner.training.engine import train
    from eeg_captioner.evaluation.run import run

    qwen_path = tmp_path / "qwen"
    Qwen2_5OmniThinkerForConditionalGeneration(tiny_config()).save_pretrained(qwen_path)
    backend = BackendTokenizer(WordLevel({"[PAD]": 0, "[UNK]": 1, "<|im_end|>": 2,
                                        "<|im_start|>": 3, "delta": 4, "alpha": 5}, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, pad_token="[PAD]", unk_token="[UNK]",
                                       eos_token="<|im_end|>", additional_special_tokens=["<|im_start|>"])
    tokenizer.save_pretrained(qwen_path)
    args = dict(d_model=256, spatial_layers=1, encoder_layers=1, dropout=0.0)
    encoder_path = tmp_path / "encoder.pt"
    torch.save({"architecture": "dualdomain_spatial", "args": args,
                "encoder": DualDomainSpatialEncoder(**args).state_dict()}, encoder_path)
    questions = read_jsonl(Path(eeg_captioner.__file__).parent / "data/questions_25.jsonl")
    rows = []
    for split in ("train", "val", "test"):
        count = 5 if split == "train" else 1
        make_h5(tmp_path / (split + ".h5"), trials=count)
        for i in range(count):
            rows.append({"id": f"{split}{i}", "eeg_path": split + ".h5", "summary_path": "unused.md",
                         "trial_index": i, "start_sample": 0, "split": split,
                         "caption": "Synthetic alpha and delta activity.",
                         "labels": [i % len(q["options"]) for q in questions]})
    manifest = tmp_path / "manifest.jsonl"; write_jsonl(manifest, rows)
    previous = None
    for stage_name, fraction, steps in (("qa", 0.0, 1), ("joint025", 0.25, 4),
                                         ("joint075", 0.75, 4), ("joint090", 0.90, 10)):
        output = tmp_path / stage_name
        config = {"stage": "qa" if stage_name == "qa" else "joint", "manifest": str(manifest),
                  "qwen_path": str(qwen_path), "encoder_checkpoint": str(encoder_path),
                  "output_dir": str(output), "device": "cpu", "dtype": "float32", "seed": 42,
                  "batch_size": 1, "gradient_accumulation_steps": 1, "max_steps": steps,
                  "caption_fraction": fraction, "learning_rate": 5e-5,
                  "eval_steps": steps, "eval_per_class": 1, "caption_eval_cases": 1,
                  "eval_batch_size": 16, "qa_max_new_tokens": 2, "caption_max_new_tokens": 2,
                  "cosine_decay": stage_name == "joint090"}
        if previous is not None:
            config["initial_checkpoint"] = str(previous)
        train(config)
        assert json.loads((output / "COMPLETED.json").read_text())["steps"] == steps
        assert (output / "best/qwen_lora/adapter_model.safetensors").is_file()
        previous = output / "final"
    report = run({"manifest": str(manifest), "qwen_path": str(qwen_path),
                  "encoder_checkpoint": str(encoder_path), "checkpoint": str(previous),
                  "device": "cpu", "dtype": "float32", "output_dir": str(tmp_path / "evaluation"),
                  "split": "test", "per_class": 1, "caption_cases": 1, "batch_size": 16,
                  "qa_max_new_tokens": 2, "caption_max_new_tokens": 2})
    assert report["n"] == 25 and report["caption"]["examples"] == 1
    assert report["split"] == "test" and not report["all_heads_and_classes_present"]
