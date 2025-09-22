"""Contains the implementations of loss utilities"""

from functools import partial

import jax
from jax import numpy as jnp, tree_util


@partial(jax.jit, static_argnames=['apply_fn'])
def cross_entropy(params, x, y, apply_fn, reg: float = 0.0):
    """Computes the cross-entropy loss of a model with L2 regularization"""
    logits = apply_fn(params, x)
    ce_loss = -jnp.mean(jnp.sum(y * jax.nn.log_softmax(logits), axis=1))
    # flatten params tree and compute L2 norm squared
    l2_norm_sq = tree_util.tree_reduce(lambda acc, leaf: acc + jnp.sum(leaf**2), params, 0.0)
    reg_loss = 0.5 * reg * l2_norm_sq
    return ce_loss + reg_loss


@jax.jit
def cross_entropy_generic(f_x, y):
    """Computes the cross-entropy loss of a kernel ridge regression model"""
    return -jnp.mean(jnp.sum(y * jax.nn.log_softmax(f_x), axis=1))


@jax.jit
def mean_squared_error(y_true, y_pred):
    """Computes the mean squared error loss"""
    return jnp.mean((y_true - y_pred)**2)