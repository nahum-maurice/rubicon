"""
A generic implementation of a parametrizable convolutional neural networks.
"""

from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from functools import partial
from typing import Any

from jax import grad, jit, random
import jax.numpy as jnp
from neural_tangents import stax
import optax

from rubicon.utils import cross_entropy, accuracy


@dataclass
class ConvNetConfig:
    """Configurations for convolutional neural networks."""
    conv_filters: list[int] = None
    kernel_sizes: list[tuple[int, int]] = None
    dense_sizes: list[int] = None
    # params_scale: float = 0.1
    learning_rate: float = 1e-3
    num_epochs: int = 10
    batch_size: int = 128
    pool_after: list[bool] = None
    pool_size: tuple[int, int] = (2, 2)
    regularization: float = 1e-6
    seed: int = 42
    activation_function: Any = stax.Relu
    input_shape: tuple[int, ...] = None
    optimizer: optax.GradientTransformationExtraArgs = optax.adam

    def __post_init__(self):
        if self.conv_filters is None: self.conv_filters = [32, 64]
        if self.kernel_sizes is None: self.kernel_sizes = [(3, 3), (3, 3)]
        if self.dense_sizes is None: self.dense_sizes = [128, 10]
        if self.pool_after is None:
            self.pool_after = [True] * len(self.conv_filters)
        assert len(self.conv_filters) == len(self.kernel_sizes), \
            "conv_filters and kernel_sizes must have the same length"
        assert len(self.pool_after) == len(self.conv_filters), \
            "pool_after and conv_filters must have the same length"
        self.num_conv_layers = len(self.conv_filters)
        self.num_dense_layers = len(self.dense_sizes)


@dataclass
class TrainingHistory:
    """Records the training history of a ConvNet."""
    epochs: list[int] = field(default_factory=list)
    train_loss: list[float] = field(default_factory=list)
    train_accuracy: list[float] = field(default_factory=list)
    test_loss: list[float] = field(default_factory=list)
    test_accuracy: list[float] = field(default_factory=list)


class ConvNet:
    def __init__(self, cfg: ConvNetConfig) -> None:
        self.cfg = cfg
        self.init_fn = None
        self.apply_fn = None
        self.kernel_fn = None
        self.params = None

    @property
    def initialized(self) -> bool:
        """Return whether or not the model is initialized
        
        Returns:
            bool: whether or not the model is initialized
        """
        return self.init_fn is not None \
            and self.apply_fn is not None \
            and self.kernel_fn is not None
        
    def initialize(self) -> 'ConvNet':
        """Initialize the model if not already initialized
        
        Returns:
            ConvNet: the initialized model
        """
        if self.initialized: return self

        layers = []
        # We first add the conv layers.
        for i in range(self.cfg.num_conv_layers):
            layer = stax.Conv(
                out_chan=self.cfg.conv_filters[i],
                filter_shape=self.cfg.kernel_sizes[i],
                strides=(1, 1),
                padding='SAME',
            )
            layers.append(layer)
            layers.append(self.cfg.activation_function())
            # We only add pooling if specified. Otherwise, we skip it
            # after a particular conv layer.
            if self.cfg.pool_after[i] and self.cfg.pool_size:
                layers.append(stax.AvgPool(self.cfg.pool_size))
        layers.append(stax.Flatten())
        # We then add the dense layers.
        for i in range(self.cfg.num_dense_layers):
            layer = stax.Dense(
                self.cfg.dense_sizes[i],
            )
            layers.append(layer)
            if i < self.cfg.num_dense_layers - 1:
                layers.append(self.cfg.activation_function())

        self.init_fn, self.apply_fn, self.kernel_fn = stax.serial(*layers)
        return self
    
    def __call__(self) -> 'ConvNet':
        """Initialize the model if not already initialized

        An alias to :meth:`initialize`
        
        Returns:
            ConvNet: the initialized model
        """
        return self.initialize()

    def train(
        self,
        data_factory: Callable[[], tuple[Iterator[tuple[jnp.ndarray, jnp.ndarray]], Iterator[tuple[jnp.ndarray, jnp.ndarray]]]],
        return_metrics: bool = False,
        verbose: bool = False
    ) -> TrainingHistory | None:
        """Trains the model using mini-batch gradient descent

        Note: The model must be initialized before traiing. If not priorly
        initialized, it will be initialized automatically before training.

        Args:
            train_iter: Iterator yielding batches of (images, labels) 
                        for training.
            train_iter_factory: Callable yielding batches of (images, labels)
                                for training. Called once per epoch to ensure
                                fresh shuffling.
            test_iter: Iterator yielding batches of (images, labels) for
                       testing.
            return_metrics: Whether to return the training metrics.
            verbose: Whether to output the training progression information.

        Returns:
            TrainingHistory containing epoch-wise metrics if return_metrics is
            true, otherwise None.
        """
        if not self.initialized: self.initialize()
        
        key = random.key(self.cfg.seed)
        _, self.params = self.init_fn(key, input_shape=self.cfg.input_shape)

        grad_loss = jit(grad(lambda p, x, y: cross_entropy(p, x, y, self.apply_fn, self.cfg.regularization)))
        optimizer = self.cfg.optimizer(learning_rate=self.cfg.learning_rate)
        opt_state = optimizer.init(self.params)
        
        training_history = TrainingHistory()
        for epoch in range(self.cfg.num_epochs):
            train_iter, test_iter = data_factory()

            train_loss_sum, train_acc_sum, num_batches = 0.0, 0.0, 0

            for x_batch, y_batch in train_iter:
                grads = grad_loss(self.params, x_batch, y_batch)
                updates, opt_state = optimizer.update(grads, opt_state)
                self.params = optax.apply_updates(self.params, updates)

                if return_metrics or verbose:
                    batch_loss = cross_entropy(self.params, x_batch, y_batch, self.apply_fn, self.cfg.regularization)
                    batch_preds = self.apply_fn(self.params, x_batch)
                    batch_accuracy = accuracy(batch_preds, y_batch)
                    train_loss_sum += batch_loss
                    train_acc_sum += batch_accuracy
                    num_batches += 1
            
            if num_batches > 0:
                train_loss_avg = train_loss_sum / num_batches
                train_acc_avg = train_acc_sum / num_batches
            else:
                train_loss_avg, train_acc_avg = 0.0, 0.0

            test_acc_avg = 0.0
            if (return_metrics or verbose):
                test_loss_sum, test_acc_sum, test_batches = 0.0, 0.0, 0

                for x_test, y_test in test_iter:
                    test_loss = cross_entropy(self.params, x_test, y_test, self.apply_fn, self.cfg.regularization)
                    test_preds = self.apply_fn(self.params, x_test)
                    test_acc = accuracy(test_preds, y_test)
                    test_loss_sum += test_loss
                    test_acc_sum += test_acc
                    test_batches += 1

                if test_batches > 0:
                    test_loss_avg = test_loss_sum / test_batches
                    test_acc_avg = test_acc_sum / test_batches
                else:
                    test_loss_avg, test_acc_avg = 0.0, 0.0

            if return_metrics:
                training_history.epochs.append(epoch)
                training_history.train_loss.append(train_loss_avg)
                training_history.train_accuracy.append(train_acc_avg)
                training_history.test_loss.append(test_loss_avg)
                training_history.test_accuracy.append(test_acc_avg)

            if verbose:
                print(f"Epoch {epoch}: ")
                print(f"\tTrain loss: {train_loss_avg:.4f}")
                print(f"\tTrain accuracy: {train_acc_avg:.4f}")
                if test_iter is not None:
                    print(f"\tTest loss: {test_loss_avg:.4f}")
                    print(f"\tTest accuracy: {test_acc_avg:.4f}")
        
        if return_metrics: return training_history
