"""Experiment configuration, loaded from a TOML file.

Everything that used to be a global variable in the notebook (paths,
hyperparameters, data augmentation ranges, random seed) lives here, so the code
itself never has to change between two experiments.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Augmentation:
    """Ranges of the random transformations applied to training images.

    The values mirror the ``ImageDataGenerator`` arguments of the original
    notebook. Fractions are relative to the image size, as in Keras.
    """

    rotation_degrees: float = 20.0
    zoom_range: float = 0.2
    shift_range: float = 0.2
    shear_range: float = 0.2
    horizontal_flip: bool = True


@dataclass(frozen=True)
class Config:
    """All the knobs of one experiment.

    >>> config = Config.from_dict({"image_size": 75, "epochs": 1})
    >>> config.image_size, config.epochs, config.seed
    (75, 1, 42)
    >>> config.augmentation.horizontal_flip
    True
    """

    dataset_url: str = (
        "https://raw.githubusercontent.com/paras42/Hello_World_Deep_Learning/"
        "master/Open_I_abd_vs_CXRs.zip"
    )
    dataset_directory: Path = Path("data/Open_I_abd_vs_CXRs")
    output_directory: Path = Path("outputs")
    image_size: int = 299
    batch_size: int = 5
    epochs: int = 20
    learning_rate: float = 1e-4
    seed: int = 42
    augmentation: Augmentation = Augmentation()

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> Config:
        """Build a configuration from a plain dictionary, e.g. parsed TOML.

        Keys that are absent keep their default value; unknown keys raise a
        ``TypeError`` so that a typo in the file is noticed immediately.

        >>> Config.from_dict({"epochs": 3}).epochs
        3
        >>> Config.from_dict({"epoch": 3})
        Traceback (most recent call last):
            ...
        TypeError: Config.__init__() got an unexpected keyword argument 'epoch'
        """
        values = dict(values)
        for key in ("dataset_directory", "output_directory"):
            if key in values:
                values[key] = Path(values[key])
        if "augmentation" in values:
            values["augmentation"] = Augmentation(**values["augmentation"])
        return cls(**values)


def load_config(path: Path) -> Config:
    """Read a TOML file and return the corresponding :class:`Config`."""
    with path.open("rb") as file:
        return Config.from_dict(tomllib.load(file))
