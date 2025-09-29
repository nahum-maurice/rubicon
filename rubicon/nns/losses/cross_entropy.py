"""Implementation of the Cross Entropy loss."""

from functools import partial

import jax
import jax.numpy as jnp

from ._base import LossFn


class CrossEntropyLoss(LossFn):
    def __call__(self, y_true: jnp.ndarray, y_pred: jnp.ndarray) -> float:
        return self.cross_entropy(y_true, y_pred)

    @partial(jax.jit, static_argnums=(0,))
    def cross_entropy(self, y_true: jnp.ndarray, y_pred: jnp.ndarray) -> float:
        """Computes the cross-entropy loss.

        Args:
          y_true: The true values.
          y_pred: The predicted values.

        Returns:
          The cross-entropy loss.
        """
        # TODO(nahum): add L2 regularization. i don't yet know to do it
        # whether it's going to be with or without params.
        return -jnp.mean(jnp.sum(y_true * jax.nn.log_softmax(y_pred), axis=1))
