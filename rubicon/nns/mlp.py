"""A generic implementation of a parametrizable MLPs."""

from dataclasses import dataclass, field
from typing import Any

from rubicon.nns._base import (
    DataArray,
    Model,
    Prediction,
    TrainingConfig,
    TrainingHistory,
)


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


class MultiLayerPerceptron(Model):
    def __init__(self, cfg: MLPConfig) -> None:
        """
        Initialize the model.

        Args:
            cfg: The configuration for the MLP.
        """

        # TODO add dropout and batch norm. This cannot be done right now (at
        # least in the case of batch norm) because `neural_tangents` does not
        # have support for BatchNorm. In the future, I plan to replace the
        # part of it that is used (basically nt.empirical_ntk_fn) and in this
        # sense, it would be possible to use full-fledge batch norm from
        # `jax.exemple_libraries.stax.BatchNorm`.

        layers = []
        for layer in cfg.hidden_layers:
            layers.append(stax.Dense(layer.size))
            layers.append(layer.activation_fn)
        layers.append(stax.Dense(cfg.output_layer.size))

        self.init_fn, self.apply_fn, self.kernel_fn = stax.serial(*layers)
        self.params = None

    @property
    def initialized(self) -> bool:
        """Return whether or not the model is initialized"""

        attrs = [self.init_fn, self.apply_fn, self.kernel_fn, self.params]
        return all([attr is not None for attr in attrs])

    def __call__(self, input_shape: tuple[int, ...], seed: int = 42) -> None:
        """Initialize the model if not already initialized. The purpose of
        this initialization (that in principle could be done in the
        constructor) is to allow the user to change the `input_size` of the
        model.

        Args:
            input_shape: The shape of the input data.
            seed: The random seed.
        """
        key = random.key(seed)
        _, self.params = self.init_fn(key, input_shape=input_shape)

    def fit(self, config: TrainingConfig) -> TrainingHistory | None:
        """Train the model using mini-batch gradient descent.
        
        Args:
            config: The training configuration.
        """
        if not self.initialized:
            raise ValueError("Model must be initialized before training.")

    def predict(self, x: DataArray) -> Prediction: ...
