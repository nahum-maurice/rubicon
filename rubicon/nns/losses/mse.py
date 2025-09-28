"""Implementation of the Mean Square Error loss."""

from jax import numpy as jnp

from ._base import LossFn


class MSELoss(LossFn):
    def __call__(self, y_true: jnp.ndarray, y_pred: jnp.ndarray) -> float:
        return self.mse(y_true, y_pred)

    @jax.jit
    def mse(self, y_true: jnp.ndarray, y_pred: jnp.ndarray) -> float:
        """Computes the mean squared error loss.

        Args:
          y_true: The true values.
          y_pred: The predicted values.

        Returns:
          The mean squared error loss.
        """
        return jnp.mean((y_true - y_pred) ** 2)
