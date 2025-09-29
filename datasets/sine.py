"""Implementation of the Sin dataset. It's just a randomly generated one,
not actually something that should be taken seriously"""

from dataclasses import dataclass
from typing import Callable, Iterator

import jax
import jax.numpy as jnp
import jax.random as random
import numpy as np


@dataclass
class SineLoaderConfig:
    batch_size: int = 32
    n_train: int = 1600  # 50 batches of 32
    n_test: int = 320  # 10 batches of 32
    noise: float = 0.1
    seed: int = 42


def load_sine(config: SineLoaderConfig) -> tuple[Iterator, Iterator]:
    """Load the sine dataset"""
    key = random.key(config.seed)

    x_train = jnp.linspace(-jnp.pi, jnp.pi, config.n_train)[:, None]
    y_train = jnp.sin(x_train) + config.noise * random.normal(
        key, shape=(config.n_train, 1)
    )
    x_test = jnp.linspace(-jnp.pi, jnp.pi, config.n_test)[:, None]
    y_test = jnp.sin(x_test) + config.noise * random.normal(
        key, shape=(config.n_test, 1)
    )

    def make_iterator(
        data_x: jnp.ndarray,
        data_y: jnp.ndarray,
        batch_size: int,
        shuffle: bool = True,
    ) -> Iterator[tuple[jnp.ndarray, jnp.ndarray]]:
        n = len(data_x)
        indices = np.arange(n)
        if shuffle:
            np.random.shuffle(indices)

        for i in range(0, n, batch_size):
            batch_indices = indices[i : i + batch_size]
            yield data_x[batch_indices], data_y[batch_indices]

    train = make_iterator(x_train, y_train, config.batch_size)
    test = make_iterator(x_test, y_test, config.batch_size, shuffle=False)
    return train, test


def create_data_factory(
    config: SineLoaderConfig,
) -> Callable[[], tuple[Iterator, Iterator]]:
    """Create a factory function for training and test iterators."""
    return lambda: load_sine(config)
