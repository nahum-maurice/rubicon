"""A generic implementation of a parametrizable MLPs."""

from dataclasses import dataclass, field
from typing import Any

from rubicon.nns._base import Model, Prediction, TrainingConfig, TrainingHistory


@dataclass
class LayerConfig:
    """Configuration for a single layer in the MLP."""

    size: int
    activation_fn: Any = stax.Relu  # TODO: fix the type
    dropout_rate: float = 0.0
    use_batch_norm: bool = False


@dataclass
class MLPConfig:
    """Configurations for multi-layer perceptrons."""

    hidden_layers: list[LayerConfig] = field(default_factory=list)
    output_layer: LayerConfig
    seed: int = 42


class MLP(Model):
    def __init__(self, cfg: MLPConfig) -> None: ...

    @property
    def initialized(self) -> bool: ...

    def initialize(self) -> "MLP": ...

    def __call__(self) -> "MLP": ...

    def train(self, config: TrainingConfig) -> TrainingHistory | None: ...

    def predict(self) -> Prediction: ...
