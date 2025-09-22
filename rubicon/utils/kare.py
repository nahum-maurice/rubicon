"""Contains the implementation of KARE utilities."""

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
    inv = jnp.linalg.inv(mat)
    inv2 = inv @ inv
    return ((1 / n) * y.T @ inv2 @ y) / ((1 / n) *jnp.trace(inv))**2
