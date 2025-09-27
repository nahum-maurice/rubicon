"""A generic implementation of a parametrizable MLPs."""

from dataclasses import dataclass, field
from typing import Any

from neural_tangents import stax
from neural_tangents._src.utils.typing import LayerKernelFn

from rubicon.common.types import DataArray
from rubicon.nns._base import (
    Model,
    Prediction,
    TrainingConfig,
    TrainingHistory,
)


@dataclass
class LayerConfig:
    """Configuration for a single layer in the MLP."""

    size: int
    activation_fn: LayerKernelFn = stax.Relu
    dropout_rate: float = 0.0
    use_batch_norm: bool = False


@dataclass
class MLPConfig:
    """Configurations for multi-layer perceptrons."""

    hidden_layers: list[LayerConfig] = field(default_factory=list)
    output_layer: LayerConfig


class MultiLayerPerceptron(Model):
    def __init__(self, cfg: MLPConfig) -> None:

        # TODO(nahum): add dropout and batch norm. This cannot be done right
        # now (at least in the case of batch norm) because `neural_tangents`
        # does not have support for BatchNorm. In the future, I plan to
        # replace the part of it that is used (basically nt.empirical_ntk_fn)
        # and in this sense, it would be possible to use full-fledge batch
        # norm from `jax.example_libraries.stax.BatchNorm`.

        layers = []
        for layer in cfg.hidden_layers:
            layers.append(stax.Dense(layer.size))
            layers.append(layer.activation_fn())
        layers.append(stax.Dense(cfg.output_layer.size))

        init, apply, kernel = stax.serial(*layers)
        super().__init__(init, apply, kernel, None)

    def fit(self, config: TrainingConfig) -> TrainingHistory | None:
        if not self.initialized:
            raise ValueError("Model must be initialized before training.")

    def predict(self, x: DataArray) -> Prediction:
        if not self.initialized:
            raise ValueError("Model must be initialized before prediction.")


__all__ = ["MultiLayerPerceptron", "MLPConfig", "LayerConfig"]
