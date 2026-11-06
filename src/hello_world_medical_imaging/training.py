"""Training a model on datasets it receives as arguments.

This module knows nothing about where the images come from: the datasets are
*injected* by the caller (dependency injection, see ``docs/adr/0001``). The
command line passes the radiographs; the tests pass synthetic images.
"""

from __future__ import annotations

import logging
from pathlib import Path

import keras
import matplotlib

matplotlib.use("Agg")  # render to files, never to a window
import matplotlib.pyplot as plt  # noqa: E402
import tensorflow as tf  # noqa: E402

logger = logging.getLogger(__name__)


def train(
    model: keras.Model,
    train_dataset: tf.data.Dataset,
    validation_dataset: tf.data.Dataset,
    epochs: int,
) -> keras.callbacks.History:
    """Fit ``model`` for ``epochs`` passes over ``train_dataset``.

    Args:
        model: A compiled Keras model.
        train_dataset: Batches of ``(images, labels)`` used to update the weights.
        validation_dataset: Batches of ``(images, labels)`` only used to report metrics.
        epochs: Number of passes over the training data.

    Returns:
        The Keras training history: one value per epoch for loss, accuracy,
        validation loss and validation accuracy.
    """
    logger.info("Training for %d epoch(s)", epochs)
    # The datasets are already shuffled (or deliberately not), hence shuffle=False.
    history: keras.callbacks.History = model.fit(
        train_dataset, epochs=epochs, validation_data=validation_dataset, shuffle=False
    )
    final = {key: values[-1] for key, values in history.history.items()}
    logger.info("Training finished: %s", final)
    return history


def plot_history(history: keras.callbacks.History, path: Path) -> Path:
    """Save the training curves (loss and accuracy, training and validation) to ``path``."""
    plt.figure()
    for key, color in (
        ("accuracy", "orange"),
        ("val_accuracy", "blue"),
        ("loss", "red"),
        ("val_loss", "green"),
    ):
        plt.plot(history.history[key], color, label=key.replace("_", " "))
    plt.xlabel("epoch")
    plt.legend()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path)
    plt.close()
    logger.info("Training curves saved to %s", path)
    return path
