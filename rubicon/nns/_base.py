"""Blueprint for a generic parametrizable model."""

from dataclasses import dataclass, field
from typing import Any

import jax
from jax import numpy as jnp, random
from jax.flatten_util import ravel_pytree
import optax

from rubicon.common.types import DataArray, DataFactory
from rubicon.nns.losses import KARELoss, MSELoss, LossFn


@dataclass
class TrainingConfig:
    """Configuration for training session."""

    num_epochs: int = 10
    batch_size: int = 128
    reg: float = 1e-6
    learning_rate: float = 1e-3
    optimizer: optax.GradientTransformationExtraArgs = optax.adam
    loss_fn: LossFn = MSELoss()

    return_metrics: bool = False
    verbose: bool = False

    data_factory: DataFactory


@dataclass
class TrainingHistory:
    """Records the training history of a ConvNet."""

    epochs: list[int] = field(default_factory=list)
    train_loss: list[float] = field(default_factory=list)
    train_accuracy: list[float] = field(default_factory=list)
    test_loss: list[float] = field(default_factory=list)
    test_accuracy: list[float] = field(default_factory=list)

    def add_training_metrics(
        self,
        epoch: int,
        train_loss: float,
        train_accuracy: float,
        test_loss: float,
        test_accuracy: float,
    ) -> None:
        """Add training metrics to the history.

        Args:
          epoch: The current epoch.
          train_loss: The training loss.
          train_accuracy: The training accuracy.
          test_loss: The test loss.
          test_accuracy: The test accuracy.
        """
        self.epochs.append(epoch)
        self.train_loss.append(train_loss)
        self.train_accuracy.append(train_accuracy)
        self.test_loss.append(test_loss)
        self.test_accuracy.append(test_accuracy)


@dataclass
class Prediction:
    """The prediction result of a model."""

    y: DataArray


class Model:
    """A base class for neural network models."""

    def __init__(
        self,
        init_fn: Any | None,  # TODO(nahum): fix the type
        apply_fn: Any | None,  # TODO(nahum): fix the type
        params: Pytree | None,
    ) -> None:
        """Create and initialize a model.

        Since this is based so far on `neural_tangents`, the elements
        `init_fn` and `apply_fn` are expected to be functions
        from `neural_tangents.stax`. The are given by calling:
        `neural_tangents.stax.serial`. The `params` is in it turn given
        by calling `init_fn` with a random key and the input shape.

        Args:
          init_fn: The initialization function.
          apply_fn: The application function.
          params: The parameters.
        """
        self.init_fn = init_fn
        self.apply_fn = apply_fn
        self.params = params

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} initialized={self.initialized}>"

    def __call__(self, input_shape: tuple[int, ...], seed: int = 42) -> None:
        """Initialize the model if not already initialized.

        The purpose of this initialization (that in principle could be done in
        the constructor) is to allow the user to change the `input_size` of
        the model.

        Args:
          input_shape: The shape of the input data.
          seed: The random seed.
        """
        if self.initialized:
            return
        key = random.key(seed)
        _, self.params = self.init_fn(key, input_shape=input_shape)

    @property
    def initialized(self):
        """Return whether or not the model is initialized.

        Returns:
          bool: whether or not the model is initialized
        """
        attrs = [self.init_fn, self.apply_fn, self.params]
        return all([attr is not None for attr in attrs])

    def fit(self, config: TrainingConfig) -> TrainingHistory | None:
        """Train the model using mini-batch gradient descent

        Args:
          config: The training configuration.

        Returns:
          TrainingHistory containing epoch-wise metrics if return_metrics is
          true, otherwise None.
        """
        if isinstance(config.loss_fn, KARELoss):
            return self._train_w_kare(config)
        return self.custom_fit(config)
    
    def _train_w_kare(self, config: TrainingConfig) -> TrainingHistory | None:
        ...

    def custom_fit(self, config: TrainingConfig) -> TrainingHistory | None:
        ...

    def predict(self, x: DataArray) -> Prediction:
        """Make predictions using the model.

        Args:
          x: The input data.

        Returns:
          Prediction: The prediction object containing the predicted values.
        """
        return self.apply_fn(self.params, x)

    def compute_gradient(self, x: jnp.ndarray) -> jnp.ndarray:
        """Compute the gradient vector with respect to the parameters of the
        model at a given point.

        Args:
          x: The input point.

        Returns:
          jnp.ndarray: The gradient vector.
        """
        assert self.initialized, "The model is not yet initialized"

        def single_output(p: Pytree, x_: jnp.ndarray):
            return self.apply_fn(p, x_.reshape(1, -1)[0, 0])

        grad_fn = jax.grad(single_output, argnums=0)

        def flat_grad(p: Pytree, x: jnp.ndarray) -> jnp.ndarray:
            g_tree = grad_fn(p, x)
            flat_g, _ = ravel_pytree(g_tree)
            return flat_g

        per_sample_grads = jax.vmap(lambda x: flat_grad(self.params, x))
        return per_sample_grads(x.squeeze())

    def compute_ntk(self, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        """Compute the NTK between two points.

        Args:
          x: The first input point.
          y: The second input point.

        Returns:
          jnp.ndarray: The NTK between the two points.
        """
        return self.compute_gradient(x) @ self.compute_gradient(y).T


__all__ = ["Model", "Prediction", "TrainingConfig", "TrainingHistory"]
