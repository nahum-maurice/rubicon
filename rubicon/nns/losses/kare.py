"""Implementation of the KARE loss."""

from functools import partial

import jax
import jax.numpy as jnp

from ._base import LossFn


class KARELoss(LossFn):
    def __call__(self, y: jnp.ndarray, K: jnp.ndarray, z: float) -> float:
        return self.kare(y, K, z)

    @partial(jax.jit, static_argnums=(0,))
    def kare(self, y: jnp.ndarray, K: jnp.ndarray, z: float) -> float:
        """Computes the Kernel Alignment Risk Estimator (KARE)

        Args:
        y: The target values.
        K: The kernel matrix.
        z: The regularization parameter.

        Returns:
        The KARE value.
        """
        n = K.shape[0]
        K_norm = K / n
        mat = K_norm + z * jnp.eye(n)
        inv = jax.jit(jnp.linalg.inv, backend="cpu")(mat)
        inv2 = inv @ inv
        return ((1 / n) * y.T @ inv2 @ y) / ((1 / n) * jnp.trace(inv)) ** 2
        # return (1 / n) * jnp.sum(y.T @ inv2 @ y) / ((1 / n) * jnp.trace(inv)) ** 2
