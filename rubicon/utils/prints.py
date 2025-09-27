"""Implementations of utility functions for printing out in various cases"""


def print_training_result(
    epoch: int,
    train_loss: float,
    train_accuracy: float,
    test_loss: float,
    test_accuracy: float,
) -> None:
    """Outputs the result of a training step.

    Args:
      epoch: The current epoch.
      train_loss: The training loss.
      train_accuracy: The training accuracy.
      test_loss: The test loss.
      test_accuracy: The test accuracy.
    """
    print(
        f"Epoch {epoch:<4} | Train loss: {train_loss:.4f} | Train accuracy: {train_accuracy:.4f} | Test loss: {test_loss:.4f} | Test accuracy: {test_accuracy:.4f}"
    )
