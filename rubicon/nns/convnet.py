"""A generic implementation of a parametrizable CNNs."""

from dataclasses import dataclass
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
from neural_tangents import stax
import optax

from rubicon.nns._base import (
    DataArray,
    Model,
    Prediction,
    TrainingConfig,
    TrainingHistory,
)


@dataclass
class ConvNetConfig:
    """Configurations for convolutional neural networks."""

    conv_filters: list[int] = None
    kernel_sizes: list[tuple[int, int]] = None
    dense_sizes: list[int] = None
    learning_rate: float = 1e-3
    pool_after: list[bool] = None
    pool_size: tuple[int, int] = (2, 2)
    reg: float = 1e-6
    seed: int = 42
    activation_fn: Any = stax.Relu  # TODO: fix the type
    input_shape: tuple[int, ...] = None
    optimizer: optax.GradientTransformationExtraArgs = optax.adam

    def __post_init__(self):
        if self.conv_filters is None:
            self.conv_filters = [32, 64]
        if self.kernel_sizes is None:
            self.kernel_sizes = [(3, 3), (3, 3)]
        if self.dense_sizes is None:
            self.dense_sizes = [128, 10]
        if self.pool_after is None:
            self.pool_after = [True] * len(self.conv_filters)
        assert len(self.conv_filters) == len(
            self.kernel_sizes
        ), "conv_filters and kernel_sizes must have the same length"
        assert len(self.pool_after) == len(
            self.conv_filters
        ), "pool_after and conv_filters must have the same length"
        self.num_conv_layers = len(self.conv_filters)
        self.num_dense_layers = len(self.dense_sizes)


class ConvolutionalNeuralNetwork(Model):
    def __init__(self, cfg: ConvNetConfig) -> None:

        # TODO(nahum): MaxPool is not supported by `neural_tangents` and this
        # is bad. In the future, I plan to replace everything stax by the
        # corresponding functions from `jax.example_libraries.stax`.
        
        layers = []
    
    def fit(self, config: TrainingConfig) -> TrainingHistory | None:
        if not self.initialized:
            raise ValueError("Model must be initialized before training.")

    def predict(self, x: DataArray) -> Prediction:
        if not self.initialized:
            raise ValueError("Model must be initialized before prediction.")


class ConvNet(Model):
    def __init__(self, cfg: ConvNetConfig) -> None:
        layers = []
        # We first add the conv layers.
        for i in range(cfg.num_conv_layers):
            layer = stax.Conv(
                out_chan=cfg.conv_filters[i],
                filter_shape=cfg.kernel_sizes[i],
                strides=(1, 1),
                padding="SAME",
            )
            layers.append(layer)
            layers.append(cfg.activation_fn())
            # We only add pooling if specified. Otherwise, we skip it
            # after a particular conv layer.
            if cfg.pool_after[i] and cfg.pool_size:
                layers.append(stax.AvgPool(cfg.pool_size))
        layers.append(stax.Flatten())
        # We then add the dense layers.
        for i in range(cfg.num_dense_layers):
            layer = stax.Dense(
                cfg.dense_sizes[i],
            )
            layers.append(layer)
            if i < cfg.num_dense_layers - 1:
                layers.append(cfg.activation_fn())

        init, apply, kernel = stax.serial(*layers)
        super().__init__(init, apply, kernel, None)

    @partial(jax.jit, static_argnames=["apply_fn"])
    def cross_entropy(self, params, x, y, apply_fn, reg: float = 0.0) -> float:
        """Compute the cross-entropy loss of a model with L2 reg

        Args:
          params: the model parameters
          x: the input data
          y: the true labels
          apply_fn: the model's forward pass function
          reg: the reg parameter

        Returns:
          float: the cross-entropy loss
        """
        logits = apply_fn(params, x)
        ce_loss = -jnp.mean(jnp.sum(y * jax.nn.log_softmax(logits), axis=1))
        # flatten params tree and compute L2 norm squared
        l2_norm_sq = tree_util.tree_reduce(
            lambda acc, leaf: acc + jnp.sum(leaf**2), params, 0.0
        )
        reg_loss = 0.5 * reg * l2_norm_sq
        return ce_loss + reg_loss


    def fit(self, config: TrainingConfig) -> TrainingHistory | None:
        if not self.initialized:
            self.initialize()

        @partial(jax.jit, static_argnames=["apply_fn"])
        def _grd_loss(p, x, y, apply_fn, reg: float = 0.0):
            return jax.grad(
                self.cross_entropy(params=p, x=x, y=y, apply_fn=apply_fn, reg=reg)
            )

        grad_loss = partial(_grd_loss, apply_fn=self.apply_fn, reg=config.reg)
        ce = partial(cross_entropy, apply_fn=self.apply_fn, reg=config.reg)
        optimizer = config.optimizer(learning_rate=config.learning_rate)
        opt_state = optimizer.init(self.params)

        training_history = TrainingHistory()
        for epoch in range(config.num_epochs):
            train_iter, test_iter = config.data_factory()
            train_loss_sum, train_acc_sum, num_batches = 0.0, 0.0, 0

            for x_batch, y_batch in train_iter:
                grads = grad_loss(self.params, x_batch, y_batch)
                updates, opt_state = optimizer.update(grads, opt_state)
                self.params = optax.apply_updates(self.params, updates)

                if config.return_metrics or config.verbose:
                    batch_loss = ce(self.params, x_batch, y_batch)
                    batch_preds = self.apply_fn(self.params, x_batch)
                    batch_accuracy = self.accuracy(batch_preds, y_batch)
                    train_loss_sum += batch_loss
                    train_acc_sum += batch_accuracy
                    num_batches += 1

            train_loss_avg = train_loss_sum / num_batches or 0.0
            train_acc_avg = train_acc_sum / num_batches or 0.0

            if config.return_metrics or config.verbose:
                test_loss_sum, test_acc_sum, test_batches = 0.0, 0.0, 0

                for x_test, y_test in test_iter:
                    test_loss = ce(self.params, x_test, y_test)
                    test_preds = self.apply_fn(self.params, x_test)
                    test_acc = self.accuracy(test_preds, y_test)
                    test_loss_sum += test_loss
                    test_acc_sum += test_acc
                    test_batches += 1

                test_loss_avg = test_loss_sum / test_batches or 0.0
                test_acc_avg = test_acc_sum / test_batches or 0.0

            if config.return_metrics:
                training_history.add_training_metrics(
                    epoch, train_loss_avg, train_acc_avg, test_loss_avg, test_acc_avg
                )
            if config.verbose:
                self.print_result(
                    epoch,
                    train_loss_avg,
                    train_acc_avg,
                    test_loss_avg,
                    test_acc_avg,
                )

        if config.return_metrics:
            return training_history

    def predict(self, x: DataArray) -> Prediction: ...
