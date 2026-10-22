"""Feature registry and shared extraction context."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Mapping, MutableMapping

import numpy as np

FeatureOutput = Dict[str, np.ndarray]
FeatureFn = Callable[[np.ndarray, "TrialContext"], FeatureOutput]


@dataclass
class TrialContext:
    sample_rate: float
    channel_names: List[str]
    trial_index: int
    trial_id: str
    source_path: str
    source_file: str
    segment_index: int = 0
    segment_start_sec: float = 0.0
    segment_duration_sec: float = 0.0
    segment_part_index: int = 0
    segment_part_name: str = ""
    segment_part_start_sec: float = 0.0
    segment_part_duration_sec: float = 0.0
    unit_id: str = ""
    bad_channel_mask: np.ndarray | None = None
    bad_trial: bool = False
    bad_trial_reason: str = ""
    cache: MutableMapping[str, object] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.unit_id:
            self.unit_id = self.trial_id


class FeatureRegistry:
    """Simple plugin registry for per-trial feature computation."""

    def __init__(self) -> None:
        self._registry: Dict[str, FeatureFn] = {}

    def register(self, name: str, fn: FeatureFn) -> None:
        if name in self._registry:
            raise ValueError(f"Feature '{name}' already registered.")
        self._registry[name] = fn

    def get(self, name: str) -> FeatureFn:
        if name not in self._registry:
            known = ", ".join(sorted(self._registry.keys()))
            raise KeyError(f"Feature '{name}' is not registered. Known: {known}")
        return self._registry[name]

    def run(
        self,
        enabled_features: Mapping[str, Mapping[str, object]],
        signal: np.ndarray,
        context: TrialContext,
    ) -> Dict[str, FeatureOutput]:
        outputs: Dict[str, FeatureOutput] = {}
        for feature_name, feature_cfg in enabled_features.items():
            if not bool(feature_cfg.get("enabled", True)):
                continue
            fn = self.get(feature_name)
            outputs[feature_name] = fn(signal, context)
        return outputs
