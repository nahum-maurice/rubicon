"""Implementation of the Mean Square Error loss."""

from functools import partial

import jax
from jax import numpy as jnp

from ._base import LossFn


class MSELoss(LossFn):
    def __call__(self, y_true: jnp.ndarray, y_pred: jnp.ndarray) -> float:
        return self.mse(y_true, y_pred)

    @partial(jax.jit, static_argnums=(0,))
    def mse(self, y_true: jnp.ndarray, y_pred: jnp.ndarray) -> float:
        """Computes the mean squared error loss.

        Args:
          y_true: The true values.
          y_pred: The predicted values.

        Returns:
          The mean squared error loss.
        """
        return 0.5 * jnp.mean((y_true - y_pred) ** 2)
