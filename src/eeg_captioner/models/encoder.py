import torch
from torch import nn
from torch.nn import functional as F
from .spectral import FixedWelchPSD

def transformer_encoder(d_model, layers, dropout):
    layer = nn.TransformerEncoderLayer(
        d_model=d_model,
        nhead=8,
        dim_feedforward=d_model * 4,
        dropout=dropout,
        activation="gelu",
        batch_first=True,
        norm_first=True,
    )
    return nn.TransformerEncoder(layer, num_layers=layers)


class SampleSharedFrequencyTokenizer(nn.Module):
    """Raw log-PSD bins with one invertible normalization per EEG sample."""

    def __init__(self, d_model=256, spatial_layers=1, dropout=0.1):
        super().__init__()
        if d_model % 8:
            raise ValueError("d-model must be divisible by 8")
        self.d_model = int(d_model)
        self.welch = FixedWelchPSD()
        self.projection = nn.Sequential(
            nn.Linear(self.welch.frequency_bins + 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.channel_embedding = nn.Embedding(20, d_model)
        self.period_embedding = nn.Embedding(3, d_model)
        self.spatial_encoder = transformer_encoder(
            d_model, spatial_layers, dropout
        )
        self.output_norm = nn.LayerNorm(d_model)

    @staticmethod
    def normalize_log_psd(log_psd):
        if log_psd.ndim != 4:
            raise ValueError("Expected log PSD [B,C,P,F]")
        mean = log_psd.mean(dim=(1, 2, 3), keepdim=True)
        std = log_psd.std(
            dim=(1, 2, 3), keepdim=True, unbiased=False
        ).clamp_min(1e-6)
        normalized = (log_psd - mean) / std
        token_mean = mean.expand(-1, log_psd.shape[1], log_psd.shape[2], 1)
        token_std = std.expand(-1, log_psd.shape[1], log_psd.shape[2], 1)
        return torch.cat((normalized, token_mean, token_std), dim=-1), mean, std

    def forward(self, eeg):
        psd = self.welch(eeg)
        log_psd = torch.log10(psd.clamp_min(1e-12))
        features, _, _ = self.normalize_log_psd(log_psd)
        tokens = self.projection(features)
        tokens = (
            tokens
            + self.channel_embedding.weight[None, :, None, :]
            + self.period_embedding.weight[None, None, :, :]
        )
        batch = tokens.shape[0]
        # Each of the three periods independently compares all 20 channels.
        period_major = tokens.permute(0, 2, 1, 3).reshape(
            batch * 3, 20, self.d_model
        )
        period_major = self.spatial_encoder(period_major)
        tokens = period_major.view(batch, 3, 20, self.d_model).permute(0, 2, 1, 3)
        return self.output_norm(tokens)


class DualDomainSpatialEncoder(nn.Module):
    """0.33-second raw patches plus pre-pooling spatial frequency tokens."""

    def __init__(
        self, d_model=256, spatial_layers=1, encoder_layers=4, dropout=0.1,
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.frequency = SampleSharedFrequencyTokenizer(
            d_model=d_model,
            spatial_layers=spatial_layers,
            dropout=dropout,
        )
        self.patch_boundaries = tuple(
            (index * 2000 // 30, (index + 1) * 2000 // 30)
            for index in range(30)
        )
        self.max_patch_samples = max(
            stop - start for start, stop in self.patch_boundaries
        )
        self.time_projection = nn.Sequential(
            nn.Linear(self.max_patch_samples, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.subpatch_embedding = nn.Embedding(10, d_model)
        self.pool_queries = nn.Parameter(torch.empty(10, d_model))
        nn.init.normal_(self.pool_queries, std=0.02)
        self.pool_attention = nn.MultiheadAttention(
            d_model, num_heads=8, dropout=dropout, batch_first=True
        )
        self.pool_norm = nn.LayerNorm(d_model)
        self.output_time_embedding = nn.Embedding(10, d_model)
        self.global_encoder = transformer_encoder(
            d_model, encoder_layers, dropout
        )
        self.output_norm = nn.LayerNorm(d_model)

    @property
    def welch(self):
        return self.frequency.welch

    def time_patches(self, eeg):
        signal = eeg.float().flatten(-2)
        patches = []
        for start, stop in self.patch_boundaries:
            patch = signal[..., start:stop]
            if patch.shape[-1] < self.max_patch_samples:
                patch = F.pad(
                    patch, (0, self.max_patch_samples - patch.shape[-1])
                )
            patches.append(patch)
        return torch.stack(patches, dim=2)

    def forward(self, eeg):
        if eeg.ndim != 4 or tuple(eeg.shape[1:]) != (20, 10, 200):
            raise ValueError(f"Expected EEG [B,20,10,200], got {tuple(eeg.shape)}")
        batch = eeg.shape[0]
        frequency = self.frequency(eeg).repeat_interleave(10, dim=2)
        time_tokens = self.time_projection(self.time_patches(eeg))
        subpatch_ids = torch.arange(10, device=eeg.device).repeat(3)
        time_tokens = time_tokens + self.subpatch_embedding(subpatch_ids)[
            None, None, :, :
        ]
        # Direct additive fusion.  Frequency tokens have already compared all
        # channels within each period before this per-channel compression.
        fused = time_tokens + frequency

        channel_memory = fused.reshape(batch * 20, 30, self.d_model)
        queries = self.pool_queries[None].expand(batch * 20, -1, -1)
        pooled, _ = self.pool_attention(
            queries, channel_memory, channel_memory, need_weights=False
        )
        pooled = self.pool_norm(pooled + queries).view(
            batch, 20, 10, self.d_model
        )
        pooled = (
            pooled
            + self.frequency.channel_embedding.weight[None, :, None, :]
            + self.output_time_embedding.weight[None, None, :, :]
        )
        memory = pooled.flatten(1, 2)
        return self.output_norm(self.global_encoder(memory))
