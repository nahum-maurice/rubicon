"""Implementation of the mean absolute error metric."""

from functools import partial

import jax
from jax import numpy as jnp

from rubicon.common.types import DataArray
from rubicon.nns.metrics._base import MetricFn


class MeanAbsoluteError(MetricFn):
    def __call__(self, preds: DataArray, true: DataArray) -> float:
        return self.mae(preds, true)

    @partial(jax.jit, static_argnums=(0,))
    def mae(self, preds: DataArray, true: DataArray) -> float:
        """Computes the mean absolute error of the model.

        Args:
          preds: The predicted values.
          true: The true values.

        Returns:
          float: The mean absolute error.
        """
        return 1 - jnp.mean(jnp.abs(preds - true))
