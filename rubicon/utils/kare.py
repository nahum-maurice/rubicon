"""Contains the implementation of KARE utilities."""

import jax
from jax import numpy as jnp


def kare(y, K, z):
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
    inv = jax.jit(jnp.linalg.inv, backend='cpu')(mat)
    inv2 = inv @ inv
    # return ((1 / n) * y.T @ inv2 @ y) / ((1 / n) *jnp.trace(inv))**2
    return (1 / n) * jnp.sum(y.T @ inv2 @ y) / ((1 / n) * jnp.trace(inv))**2
