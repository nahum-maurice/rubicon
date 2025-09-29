"""A generic implementation of a parametrizable MLPs."""

from dataclasses import dataclass, field
from typing import Any

import jax
import jax.numpy as jnp
from jax.example_libraries import stax
import optax

from rubicon.common.types import DataArray
from rubicon.nns._base import (
    Model,
    Prediction,
    TrainingConfig,
    TrainingHistory,
)
from rubicon.utils.prints import print_training_result


@dataclass
class LayerConfig:
    """Configuration for a single layer in the MLP."""

    size: int = 1
    activation_fn: Any = stax.Relu  # TODO(nahum): fix the type
    dropout_rate: float = 0.0
    use_batch_norm: bool = False


@dataclass
class MLPConfig:
    """Configurations for multi-layer perceptrons."""

    hidden_layers: list[LayerConfig] = field(default_factory=list)
    output_layer: LayerConfig = LayerConfig()


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
            layers.append(layer.activation_fn)
        layers.append(stax.Dense(cfg.output_layer.size))

        init, apply = stax.serial(*layers)
        super().__init__(init, apply, None)

    def custom_train(self, config: TrainingConfig) -> TrainingHistory | None:
        if not self.initialized:
            raise ValueError("Model must be initialized before training.")

        @jax.jit
        def compute_loss(params, x, y):
            fx = self.apply_fn(params, x)
            return config.loss_fn(y, fx)

        grad_loss = jax.grad(compute_loss)
        optimizer = config.optimizer(learning_rate=config.learning_rate)
        opt_state = optimizer.init(self.params)

        step, history = 0, TrainingHistory()
        for _ in range(config.num_epochs):
            train_iter, _ = config.data_factory()

            for x_train, y_train in train_iter:
                grads = grad_loss(self.params, x_train, y_train)
                updates, opt_state = optimizer.update(grads, opt_state)
                self.params = optax.apply_updates(self.params, updates)

                train_preds = self.apply_fn(self.params, x_train)
                train_loss = config.loss_fn(y_train, train_preds)
                train_accuracy = config.accuracy_fn(train_preds, y_train)

                _, test_iter = config.data_factory()
                test_loss_tot, test_accuracy_tot, num_examples = 0.0, 0.0, 0
                for x_test, y_test in test_iter:
                    test_preds = self.apply_fn(self.params, x_test)
                    test_loss = config.loss_fn(y_test, test_preds)
                    test_accuracy = config.accuracy_fn(test_preds, y_test)
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

    def predict(self, x: DataArray) -> Prediction:
        if not self.initialized:
            raise ValueError("Model must be initialized before prediction.")


__all__ = ["MultiLayerPerceptron", "MLPConfig", "LayerConfig"]
