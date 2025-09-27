import jax

from rubicon.common.types import DataArray
from rubicon.nns.metrics._base import MetricFn


class Accuracy(MetricFn):
    def __call__(self, preds: DataArray, true: DataArray) -> float:
        return self.accuracy(preds, true)

    @jax.jit
    def accuracy(self, preds: DataArray, true: DataArray) -> float:
        """Computes the accuracy of the model.

        Args:
          preds: The predicted labels.
          true: The true labels.

        Returns:
          float: The accuracy.
        """
        return jnp.mean(jnp.argmax(preds, axis=1) == jnp.argmax(true, axis=1))
