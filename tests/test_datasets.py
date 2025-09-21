"""Testing the utilities used to work with datasets"""

from datasets.mnist import load_mnist, MnistLoaderConfig


def test_load_mnist_dataset():
    config = MnistLoaderConfig(batch_size=128)
    train, test = load_mnist(config)
    assert len(list(train)) == 469
    assert len(list(test)) == 79
    for images, labels in list(train)[:1]:
        assert images.shape == (128, 28, 28, 1)
        assert labels.shape == (128, 10)
    for images, labels in list(test)[:1]:
        assert images.shape == (128, 28, 28, 1)
        assert labels.shape == (128, 10)