import torch
from torch import nn
import math

class FixedWelchPSD(nn.Module):
    """SciPy-compatible one-window Welch PSD for the three label periods."""

    def __init__(
        self,
        sample_rate: float = 200.0,
        nperseg: int = 666,
        frequency_low: float = 0.5,
        frequency_high: float = 45.0,
    ):
        super().__init__()
        if nperseg != 666:
            raise ValueError("The v2 feature pipeline uses nperseg=666")
        window = torch.hann_window(nperseg, periodic=True, dtype=torch.float32)
        frequencies = torch.fft.rfftfreq(nperseg, d=1.0 / sample_rate)
        selected = (frequencies >= frequency_low) & (frequencies <= frequency_high)
        self.register_buffer("window", window)
        self.register_buffer("frequency_indices", selected.nonzero().flatten())
        self.register_buffer("frequencies", frequencies[selected])
        self.sample_rate = float(sample_rate)
        self.nperseg = int(nperseg)
        self.psd_scale = float(1.0 / (sample_rate * window.square().sum().item()))

    @property
    def frequency_bins(self):
        return int(self.frequency_indices.numel())

    def forward(self, eeg):
        if eeg.ndim != 4 or tuple(eeg.shape[1:]) != (20, 10, 200):
            raise ValueError(f"Expected EEG [B,20,10,200], got {tuple(eeg.shape)}")

        # np.linspace(0, 2000, 4, dtype=int) in the source pipeline produces
        # [0, 666, 1333, 2000].  SciPy Welch uses one 666-sample window for
        # each part, so the final sample of the 667-sample parts is unused.
        signal = eeg.float().flatten(-2)
        parts = torch.stack(
            (
                signal[..., 0:666],
                signal[..., 666:1332],
                signal[..., 1333:1999],
            ),
            dim=2,
        )
        parts = parts - parts.mean(dim=-1, keepdim=True)

        # torch.fft does not accept BF16 on all supported GPUs.  The fixed
        # front-end therefore always runs in FP32 even when the learned model
        # is under autocast.
        with torch.autocast(device_type=eeg.device.type, enabled=False):
            spectrum = torch.fft.rfft(parts * self.window, dim=-1)
            psd = spectrum.abs().square() * self.psd_scale
            psd[..., 1:-1] *= 2.0
            psd = psd.index_select(-1, self.frequency_indices)
        return psd


def learning_rate_factor(step, warmup_steps, max_steps, min_ratio):
    if step < warmup_steps:
        return max((step + 1) / max(warmup_steps, 1), 1e-6)
    progress = (step - warmup_steps) / max(max_steps - warmup_steps, 1)
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))
    return min_ratio + (1.0 - min_ratio) * cosine
