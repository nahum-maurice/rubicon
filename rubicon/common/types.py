"""Common types used across the library."""

from typing import Any, Callable, Iterator

from jax import numpy as jnp
import numpy as np


DataArray = jnp.ndarray | np.ndarray
"""An array of data points that can be either a JAX array or a NumPy array."""

DataIterator = Iterator[tuple[DataArray, DataArray]]
"""An iterator that yields batches of data. Each batch is a tuple of 
(features, labels).
"""

DataFactory = Callable[[], tuple[DataIterator, DataIterator]]
"""A factory that creates a training and testing data iterator."""

PyTree = Any
"""A PyTree, see `JAX docs` for more information.

.. _JAX docs: https://jax.readthedocs.io/en/latest/pytrees.html
"""
