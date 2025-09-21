"""Contains the implementations of metrics."""

import jax
from jax import numpy as jnp


@jax.jit
def accuracy(preds, true):
    """Computes the accuracy of predictions"""
    return jnp.mean(jnp.argmax(preds, axis=1) == jnp.argmax(true, axis=1))


def precision(preds, true):
    """Of the items predicted as positive, how many are acutally positive"""
    raise NotImplementedError


def recall(preds, true):
    """Of the actual positive items, how many were correcly predicted"""
    raise NotImplementedError
