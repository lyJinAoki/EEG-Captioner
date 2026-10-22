from pathlib import Path
import torch
from .encoder import DualDomainSpatialEncoder


def load_encoder(path, device="cpu", frozen=True):
    """Read original classifier checkpoints or portable encoder exports strictly."""
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    if payload.get("architecture") != "dualdomain_spatial":
        raise ValueError("Expected architecture=dualdomain_spatial")
    args = payload["args"]
    model = DualDomainSpatialEncoder(**{
        k: args[k] for k in ("d_model", "spatial_layers", "encoder_layers", "dropout")
    })
    state = payload.get("encoder")
    if state is None:
        state = {k.removeprefix("encoder."): v for k, v in payload["model"].items() if k.startswith("encoder.")}
    if not state:
        raise ValueError("Checkpoint contains no encoder tensors")
    model.load_state_dict(state, strict=True)
    model.to(device)
    if frozen:
        model.requires_grad_(False).eval()
    return model


def export_encoder(source, destination):
    # The original trusted research checkpoint may contain pathlib metadata.
    payload = torch.load(source, map_location="cpu", weights_only=False)
    if payload.get("architecture") != "dualdomain_spatial":
        raise ValueError("Wrong encoder architecture")
    state = {k.removeprefix("encoder."): v for k, v in payload["model"].items() if k.startswith("encoder.")}
    args = {k: payload["args"][k] for k in ("d_model", "spatial_layers", "encoder_layers", "dropout")}
    Path(destination).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"architecture": "dualdomain_spatial", "args": args, "encoder": state,
                "source_step": int(payload["step"])}, destination)
    load_encoder(destination)

