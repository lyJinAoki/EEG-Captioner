"""Export a relocatable final checkpoint without copying Qwen base weights."""
from pathlib import Path
import json
import shutil
from eeg_captioner.io import sha256, write_json
from eeg_captioner.models.checkpoint import export_encoder


def export_bundle(encoder, adapter, destination):
    encoder, adapter, destination = map(Path, (encoder, adapter, destination))
    if destination.exists():
        raise FileExistsError(destination)
    for path in [encoder, adapter / "eeg_projector.pt", adapter / "qwen_lora/adapter_model.safetensors", adapter / "qwen_lora/adapter_config.json"]:
        if not path.is_file():
            raise FileNotFoundError(path)
    destination.mkdir(parents=True)
    export_encoder(encoder, destination / "encoder.pt")
    shutil.copy2(adapter / "eeg_projector.pt", destination / "eeg_projector.pt")
    (destination / "qwen_lora").mkdir()
    for name in ["adapter_model.safetensors", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "special_tokens_map.json"]:
        source = adapter / "qwen_lora" / name
        if source.exists():
            shutil.copy2(source, destination / "qwen_lora" / name)
    config = json.loads((adapter / "qwen_lora/adapter_config.json").read_text())
    config["base_model_name_or_path"] = "Qwen/Qwen2.5-Omni-7B"
    write_json(destination / "qwen_lora/adapter_config.json", config)
    paths = sorted(p for p in destination.rglob("*") if p.is_file())
    write_json(destination / "bundle.json", {
        "schema": 1, "architecture": "dualdomain_spatial", "task": "joint",
        "base_model": "Qwen/Qwen2.5-Omni-7B", "eeg_units": "microvolts", "model_scale": 100,
        "source_encoder_sha256": sha256(encoder),
        "source_adapter_sha256": sha256(adapter / "qwen_lora/adapter_model.safetensors"),
        "files": {str(p.relative_to(destination)): sha256(p) for p in paths}})
    return destination

