"""Implementation of the accuracy metric."""

from functools import partial

import jax
from jax import numpy as jnp

from rubicon.common.types import DataArray
from rubicon.nns.metrics._base import MetricFn


class ClassificationAccuracy(MetricFn):
    def __call__(self, preds: DataArray, true: DataArray) -> float:
        return self.accuracy(preds, true)

    
    @partial(jax.jit, static_argnums=(0,))
    def accuracy(self, preds: DataArray, true: DataArray) -> float:
        """Computes the accuracy of the model.

        Args:
          preds: The predicted labels.
          true: The true labels.

        Returns:
          float: The accuracy.
        """
        return jnp.mean(jnp.argmax(preds, axis=1) == jnp.argmax(true, axis=1))
