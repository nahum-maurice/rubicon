"""Blueprint for a generic parametrizable model."""

from dataclasses import dataclass, field
from typing import Any, Callable, Iterator

import jax
from jax import numpy as jnp
from neural_tangents._src.utils.typing import InitFn, ApplyFn, KernelFn
import numpy as np
import optax


DataArray = jnp.ndarray | np.ndarray
DataIterator = Iterator[tuple[DataArray, DataArray]]
DataFactory = Callable[[], tuple[DataIterator, DataIterator]]

PyTree = Any
"""A PyTree, see `JAX docs` for more information.

.. _JAX docs: https://jax.readthedocs.io/en/latest/pytrees.html
"""

Params = PyTree
"""The model parameters."""


@dataclass
class TrainingConfig:
    """Configuration for training session."""

    num_epochs: int = 10
    batch_size: int = 128
    reg: float = 1e-6
    learning_rate: float = 1e-3
    optimizer: optax.GradientTransformationExtraArgs = optax.adam

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
        init_fn: InitFn | None,
        apply_fn: ApplyFn | None,
        kernel_fn: KernelFn | None,
        params: Params | None,
    ) -> None:
        """Create and initialize a model.

        Since this is based so far on `neural_tangents`, the elements
        `init_fn`, `apply_fn` and `kernel_fn` are expected to be functions
        from `neural_tangents.stax`. The are given by calling:
        `neural_tangents.stax.serial`. The `params` is in it turn given
        by calling `init_fn` with a random key and the input shape.

        Args:
            init_fn: The initialization function.
            apply_fn: The application function.
            kernel_fn: The kernel function.
            params: The parameters.
        """
        self.init_fn = init_fn
        self.apply_fn = apply_fn
        self.kernel_fn = kernel_fn
        self.params = params

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} initialized={self.initialized}>"

    def __call__(self, *args, **kwargs) -> None:
        """Initialize the model if not already initialized.

        The purpose of this initialization (that in principle could be done in
        the constructor) is to allow the user to change the `input_size` of
        the model.

        Args:
            input_shape: The shape of the input data.
            seed: The random seed.
        """
        raise NotImplementedError

    @property
    def initialized(self):
        """Return whether or not the model is initialized.

        Returns:
            bool: whether or not the model is initialized
        """
        raise NotImplementedError

    def fit(self, config: TrainingConfig) -> TrainingHistory | None:
        """Train the model using mini-batch gradient descent

        Args:
            config: The training configuration.

        Returns:
            TrainingHistory containing epoch-wise metrics if return_metrics is
            true, otherwise None.
        """
        raise NotImplementedError

    def predict(self, x: DataArray) -> Prediction:
        raise NotImplementedError

    def print_result(
        self, epoch: int, train_l: float, train_a: float, test_l: float, test_a: float
    ) -> None:
        """Outputs the result of a training step.

        Args:
            epoch: The current epoch.
            train_l: The training loss.
            train_a: The training accuracy.
            test_l: The test loss.
            test_a: The test accuracy.
        """
        print(
            f"Epoch {epoch:<4} | Train loss: {train_l:.4f} | "
            f"Train accuracy: {train_a:.4f} | "
            f"Test loss: {test_l:.4f} | "
            f"Test accuracy: {test_a:.4f}"
        )

    @jax.jit
    def accuracy(self, preds, true) -> float:
        """Computes the accuracy of the model.

        Args:
            preds: The predicted labels.
            true: The true labels.

        Returns:
            float: The accuracy.
        """
        return jnp.mean(jnp.argmax(preds, axis=1) == jnp.argmax(true, axis=1))
