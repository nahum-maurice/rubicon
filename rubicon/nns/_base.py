"""Blueprint for a generic parametrizable model."""

from dataclasses import dataclass, field
from functools import partial
from typing import Any

import jax
from jax import numpy as jnp, random
from jax.flatten_util import ravel_pytree
import optax

from rubicon.common.types import DataArray, DataFactory, PyTree
from rubicon.nns.losses import KARELoss, MSELoss, LossFn
from rubicon.nns.metrics import MetricFn, MeanAbsoluteError
from rubicon.utils.prints import print_training_result


@dataclass
class TrainingConfig:
    """Configuration for training session."""

    data_factory: DataFactory

    num_epochs: int = 10
    batch_size: int = 128
    reg: float = 1e-6
    learning_rate: float = 1e-3
    optimizer: optax.GradientTransformationExtraArgs = optax.adam
    loss_fn: LossFn = MSELoss()
    accuracy_fn: MetricFn = MeanAbsoluteError()

    verbose: bool = False

    with_kare: bool = False


@dataclass
class NTKTrainingConfig(TrainingConfig):
    """Additional configurations for training session using NTK."""

    z: float = 1e-3
    lambd: float = 1e-6
    update_params: bool = False


@dataclass
class TrainingHistory:
    """Records the training history of a ConvNet."""

    steps: list[int] = field(default_factory=list)
    train_loss: list[float] = field(default_factory=list)
    train_accuracy: list[float] = field(default_factory=list)
    test_loss: list[float] = field(default_factory=list)
    test_accuracy: list[float] = field(default_factory=list)

    def add_training_metrics(
        self,
        step: int,
        train_loss: float,
        train_accuracy: float,
        test_loss: float,
        test_accuracy: float,
    ) -> None:
        """Add training metrics to the history.

        Args:
          step: The current step.
          train_loss: The training loss.
          train_accuracy: The training accuracy.
          test_loss: The test loss.
          test_accuracy: The test accuracy.
        """
        self.steps.append(step)
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
        params: PyTree | None,
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
          TrainingHistory containing step-wise metrics
        """
        if config.with_kare:
            return self._train_w_kare(config)
        return self.custom_train(config)

    def _train_w_kare(self, config: TrainingConfig) -> TrainingHistory | None:
        """Train the model with KARE loss to obtain the NTK-KARE.

        Args:
          config: The training configuration.

        Returns:
          TrainingHistory containing step-wise metrics
        """

        @jax.jit
        def kare_loss(params: PyTree, x, y, z,):
            K = self.compute_ntk(params, x, x)
            _loss_fn = KARELoss()
            return _loss_fn(y, K, z)

        grad_kare = jax.grad(kare_loss)
        optimizer = config.optimizer(config.learning_rate)
        # NOTE: we are using `self.params` which means we are starting the
        # optimization from the current parameters of the model. In the future
        # we should allow to specify a checkpoint to load the parameters from
        # and to make this optional.
        params = self.params
        opt_state = optimizer.init(params)

        step, history = 0, TrainingHistory()
        for _ in range(config.num_epochs):
            train_iter, _ = config.data_factory()

            for x, y in train_iter:
                grads = grad_kare(params, x, y, config.z)
                updates, opt_state = optimizer.update(grads, opt_state)
                params = optax.apply_updates(params, updates)

                K = self.compute_ntk(params, x, x)

                train_preds = self.ntk_predict(params, K, x, x, y, config.lambd)
                train_loss = config.loss_fn(y, train_preds)
                train_accuracy = config.accuracy_fn(y, train_preds)

                _, test_iter = config.data_factory()
                test_loss_tot, test_accuracy_tot, num_examples = 0.0, 0.0, 0
                for x_test, y_test in test_iter:
                    test_preds = self.ntk_predict(params, K, x, x_test, y, config.lambd)
                    test_loss = config.loss_fn(y_test, test_preds)
                    test_accuracy = config.accuracy_fn(y_test, test_preds)
                    test_loss_tot += test_loss * y_test.shape[0]
                    test_accuracy_tot += test_accuracy * y_test.shape[0]
                    num_examples += y_test.shape[0]
                test_loss = test_loss_tot / num_examples
                test_accuracy = test_accuracy_tot / num_examples

                history.add_training_metrics(
                    step,
                    train_loss,
                    train_accuracy,
                    test_loss,
                    test_accuracy,
                )
                step += 1

                if step % 10 == 0:
                    if config.verbose:
                        print_training_result(
                            step,
                            train_loss,
                            train_accuracy,
                            test_loss,
                            test_accuracy,
                        )
        return history

    def custom_train(self, config: TrainingConfig) -> TrainingHistory | None:
        raise NotImplementedError

    def predict(self, x: DataArray) -> Prediction:
        """Make predictions using the model.

        Args:
          x: The input data.

        Returns:
          Prediction: The prediction object containing the predicted values.
        """
        return self.apply_fn(self.params, x)

    def ntk_predict(
        self,
        params: PyTree,
        K_train: jnp.ndarray,
        x_train: jnp.ndarray,
        x_test: jnp.ndarray,
        y_train: jnp.ndarray,
        lambd: float = 1e-6,
    ) -> Prediction:
        """Make prediction using the NTK kernel.

        Args:
          K_train: The trained Kernel K(X, X)
          x_train: The training data
          x_test: The test data
          y_train: The training labels
          params: The model parameters
          lambd: The regularization parameter

        Returns:
          Prediction: The prediction object containing the predicted values.
        """
        n = K_train.shape[0]
        K_mixed = self.compute_ntk(params, x_train, x_test)
        inv = jnp.linalg.inv((1 / n) * K_train + lambd * jnp.eye(n))
        return (1 / n) * K_mixed @ inv @ y_train

    @partial(jax.jit, static_argnums=(0,))
    def compute_gradient(self, params: PyTree, xs: jnp.ndarray) -> jnp.ndarray:
        """Compute the gradient vector with respect to the parameters of the
        model at a given point.

        Args:
          x: The input point.
          params: The model parameters.

        Returns:
          jnp.ndarray: The gradient vector.
        """
        assert self.initialized, "The model is not yet initialized"

        grad_fn = jax.grad(lambda p, x: self.apply_fn(p, x).squeeze())

        def _pointwise(x):
            flat_grads = []
            for layer in grad_fn(params, x):
                for part in layer:
                    flat_grads.append(part.flatten())
            return jnp.concatenate(flat_grads)
        
        per_sample = jax.vmap(_pointwise, in_axes=0)
        return per_sample(xs)

    @partial(jax.jit, static_argnums=(0,))
    def compute_ntk(
        self,
        params: PyTree,
        xs1: jnp.ndarray,
        xs2: jnp.ndarray,
    ) -> jnp.ndarray:
        """Compute the NTK between a batch of pairs of points.

        Args:
          xs1: The first input point.
          xs2: The second input point.

        Returns:
          jnp.ndarray: The NTK between the two points.
        """
        G1 = self.compute_gradient(params, xs1)
        G2 = self.compute_gradient(params, xs2)
        return G1.dot(G2.T)


__all__ = ["Model", "Prediction", "TrainingConfig", "TrainingHistory"]
