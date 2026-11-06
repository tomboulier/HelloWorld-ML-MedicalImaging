"""Loading the radiographs.

Two sources are provided, with the same contract: a pair of
``(training, validation)`` datasets whose elements are batches of RGB images
with pixel values in ``[0, 255]`` and binary labels (``0`` for chest, ``1`` for
abdomen). Because :func:`hello_world_medical_imaging.training.train` only
depends on that contract, one source can replace the other without touching
the model or the training code.
"""

from __future__ import annotations

import logging
import tempfile
import urllib.request
import zipfile
from pathlib import Path

import keras
import numpy as np
import tensorflow as tf

logger = logging.getLogger(__name__)

#: Sub-directories of the dataset, as shipped in the tutorial's zip archive.
TRAIN_SUBDIRECTORY = "TRAIN"
VALIDATION_SUBDIRECTORY = "VAL"
TEST_SUBDIRECTORY = "TEST"

DatasetPair = tuple[tf.data.Dataset, tf.data.Dataset]


def download_dataset(url: str, destination: Path) -> Path:
    """Download and unpack the dataset archive, unless ``destination`` exists.

    The archive is expected to contain a single top-level directory named like
    ``destination``, with ``TRAIN``, ``VAL`` and ``TEST`` sub-directories.

    Args:
        url: Where to fetch the zip archive from (``http(s)://`` or ``file://``).
        destination: Directory that will hold the unpacked images.

    Returns:
        ``destination``, for convenience.
    """
    if destination.exists():
        logger.info("Dataset already present in %s, skipping download", destination)
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading %s", url)
    with tempfile.NamedTemporaryFile(suffix=".zip") as archive:
        urllib.request.urlretrieve(url, archive.name)
        with zipfile.ZipFile(archive.name) as zipped:
            zipped.extractall(destination.parent)
    if not destination.exists():
        raise FileNotFoundError(f"Archive from {url} did not contain {destination.name}")
    logger.info("Dataset unpacked in %s", destination)
    return destination


def load_image_datasets(
    directory: Path, image_size: int, batch_size: int, seed: int
) -> DatasetPair:
    """Read the ``TRAIN`` and ``VAL`` image folders as Keras datasets.

    Each folder holds one sub-directory per class; class indices follow the
    alphabetical order of the sub-directory names.

    Args:
        directory: Root of the dataset (the one containing ``TRAIN`` and ``VAL``).
        image_size: Images are resized to ``image_size`` x ``image_size`` pixels.
        batch_size: Number of images per batch.
        seed: Seed of the shuffling, for reproducible epochs.
    """

    def read(subdirectory: str, shuffle: bool) -> tf.data.Dataset:
        dataset: tf.data.Dataset = keras.utils.image_dataset_from_directory(
            directory / subdirectory,
            label_mode="binary",
            image_size=(image_size, image_size),
            batch_size=batch_size,
            shuffle=shuffle,
            seed=seed,
        )
        return dataset

    return read(TRAIN_SUBDIRECTORY, shuffle=True), read(VALIDATION_SUBDIRECTORY, shuffle=False)


def synthetic_datasets(
    image_size: int, batch_size: int, seed: int, train_samples: int = 8, validation_samples: int = 4
) -> DatasetPair:
    """Generate random images with random labels, with the same contract as the real data.

    This second data source exists to exercise the training pipeline in
    seconds, with no download and no real patient data: the images are noise
    and the model cannot learn anything from them.

    >>> train, validation = synthetic_datasets(image_size=32, batch_size=4, seed=0)
    >>> images, labels = next(iter(train))
    >>> tuple(images.shape), tuple(labels.shape)
    ((4, 32, 32, 3), (4, 1))
    """
    generator = np.random.default_rng(seed)

    def make(samples: int) -> tf.data.Dataset:
        images = generator.integers(0, 256, size=(samples, image_size, image_size, 3))
        labels = generator.integers(0, 2, size=(samples, 1))
        return tf.data.Dataset.from_tensor_slices(
            (images.astype(np.float32), labels.astype(np.float32))
        ).batch(batch_size)

    return make(train_samples), make(validation_samples)
