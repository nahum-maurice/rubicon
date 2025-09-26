"""MNIST dataset loader"""

from collections.abc import Iterator
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import tensorflow as tf
import tensorflow_datasets as tfds


@dataclass
class MnistLoaderConfig:
    batch_size: int = 128
    num_class: int = 10


def preprocess(
    image, label, num_class: int = 10
) -> tuple[jnp.array, jnp.array]:
    """Preprocess the image and label.

    The image is normalized to [-1, 1] and the label is one-hot encoded.

    Args:
        image: The image to preprocess.
        label: The label to preprocess.
        num_class: The number of classes for the labels.

    Returns:
        A tuple of the preprocessed image and label.
    """
    image = jnp.array(image, dtype=jnp.float32) / 255.0 * 2 - 1
    label = jax.nn.one_hot(label, num_class)
    return image, label


def load_mnist(
    config: MnistLoaderConfig = MnistLoaderConfig(),
) -> tuple[Iterator, Iterator]:
    """Load the MNIST dataset.

    Args:
        config: The configuration for the dataset loader.

    Returns:
        A tuple of two iterators, one for the training set and one for the test set.
    """
    ds_train, ds_test = tfds.load(
        "mnist:3.*.*",
        split=("train", "test"),
        as_supervised=True,
    )
    ds_train = (
        ds_train.cache()
        .shuffle(1000)
        .batch(config.batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )
    ds_test = (
        ds_test.cache()
        .shuffle(1000)
        .batch(config.batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    train = map(lambda item: preprocess(*item), ds_train.as_numpy_iterator())
    test = map(lambda item: preprocess(*item), ds_test.as_numpy_iterator())

    return train, test
