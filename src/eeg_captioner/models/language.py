from pathlib import Path
import torch
from .checkpoint import load_encoder
from .projector import AtomicEEGProjector
from .prompts import QA_SYSTEM, JOINT_SYSTEM, CAPTION_QUESTION
from .language_ops import generate_batch

LORA_TARGETS = r"^model\.layers\.\d+\.self_attn\.(q_proj|k_proj|v_proj|o_proj)$"


def load_language(qwen_path, encoder_path, adapter=None, device="cpu", trainable=False,
                  task="joint", dtype="bfloat16", local_files_only=True):
    from transformers import AutoTokenizer, Qwen2_5OmniThinkerForConditionalGeneration
    from peft import PeftModel, LoraConfig, get_peft_model
    if dtype not in {"float32", "bfloat16", "float16"}:
        raise ValueError("Unsupported dtype")
    encoder = load_encoder(encoder_path, device=device)
    tokenizer = AutoTokenizer.from_pretrained(qwen_path, local_files_only=local_files_only)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
        qwen_path, torch_dtype=getattr(torch, dtype), local_files_only=local_files_only,
    ).to(device)
    base.requires_grad_(False)
    if adapter is not None:
        qwen = PeftModel.from_pretrained(base, str(Path(adapter) / "qwen_lora"), is_trainable=trainable)
    else:
        qwen = get_peft_model(base, LoraConfig(task_type="CAUSAL_LM", r=8, lora_alpha=32,
                                            lora_dropout=0.05, target_modules=LORA_TARGETS))
    qwen._eeg_system_prompt = QA_SYSTEM if task == "qa" else JOINT_SYSTEM
    projector = AtomicEEGProjector(base.config.text_config.hidden_size).to(device)
    if adapter is not None:
        projector.load_state_dict(torch.load(Path(adapter) / "eeg_projector.pt",
                                             map_location="cpu", weights_only=True), strict=True)
    if trainable:
        qwen.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    else:
        qwen.requires_grad_(False).eval()
        projector.requires_grad_(False).eval()
    return encoder, projector, qwen, tokenizer


class Captioner:
    def __init__(self, qwen_path, encoder_path, checkpoint, device="cpu", dtype="float32", task="joint"):
        self.device = torch.device(device)
        self.encoder, self.projector, self.qwen, self.tokenizer = load_language(
            qwen_path, encoder_path, checkpoint, device, False, task, dtype)

    @torch.inference_mode()
    def generate(self, eeg, question=CAPTION_QUESTION, max_new_tokens=320):
        if eeg.ndim == 3:
            eeg = eeg.unsqueeze(0)
        batch = {"x": eeg, "question": [question] * len(eeg)}
        return generate_batch(batch, self.encoder, self.projector, self.qwen,
                              self.tokenizer, self.device, max_new_tokens)

