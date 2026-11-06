"""Command-line entry point: download, train, save, predict.

This is the only module that knows about every other one. It is the
*composition root*: it reads the configuration, chooses the data source and
the model, and hands them to the training function.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import keras

from hello_world_medical_imaging.config import Config, load_config
from hello_world_medical_imaging.data import (
    TEST_SUBDIRECTORY,
    download_dataset,
    load_image_datasets,
)
from hello_world_medical_imaging.model import build_model
from hello_world_medical_imaging.predict import label_from_score, predict_image
from hello_world_medical_imaging.training import plot_history, train

logger = logging.getLogger(__name__)


def parse_args(argv: list[str]) -> argparse.Namespace:
    """Parse the command line.

    >>> parse_args([]).config
    PosixPath('config.toml')
    >>> parse_args(["--config", "other.toml"]).config
    PosixPath('other.toml')
    """
    parser = argparse.ArgumentParser(
        prog="hello-world-medical-imaging",
        description="Train a chest-versus-abdomen radiograph classifier.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config.toml"),
        help="TOML file with paths, hyperparameters and the random seed (default: %(default)s)",
    )
    return parser.parse_args(argv)


def run(config: Config) -> None:
    """Run the whole experiment described by ``config``."""
    keras.utils.set_random_seed(config.seed)

    dataset_directory = download_dataset(config.dataset_url, config.dataset_directory)
    train_dataset, validation_dataset = load_image_datasets(
        dataset_directory, config.image_size, config.batch_size, config.seed
    )
    model = build_model(config.image_size, config.learning_rate, config.augmentation)
    history = train(model, train_dataset, validation_dataset, config.epochs)

    config.output_directory.mkdir(parents=True, exist_ok=True)
    model.save(config.output_directory / "model.keras")
    plot_history(history, config.output_directory / "training_curves.png")

    for image_path in sorted((dataset_directory / TEST_SUBDIRECTORY).glob("*.png")):
        score = predict_image(model, image_path, config.image_size)
        logger.info("%s: score %.3f, predicted %s", image_path.name, score, label_from_score(score))


def main(argv: list[str] | None = None) -> None:
    """Entry point of the ``hello-world-medical-imaging`` command."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    args = parse_args(sys.argv[1:] if argv is None else argv)
    run(load_config(args.config))


if __name__ == "__main__":
    main()
