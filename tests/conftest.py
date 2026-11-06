"""Fixtures shared by the tests."""

from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

#: Small enough for Inception V3 (its minimum is 75) and fast on a laptop CPU.
IMAGE_SIZE = 75


def write_random_png(path: Path, size: int, seed: int) -> None:
    """Write a random RGB image, as a stand-in for a radiograph."""
    pixels = np.random.default_rng(seed).integers(0, 256, size=(size, size, 3), dtype=np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(pixels).save(path)


@pytest.fixture
def image_directory(tmp_path: Path) -> Path:
    """A dataset folder with the layout of the tutorial's archive, filled with noise."""
    root = tmp_path / "Open_I_abd_vs_CXRs"
    for subdirectory, count in (("TRAIN", 4), ("VAL", 2)):
        for class_name in ("openI_CXR", "openI_abd_xray"):
            for index in range(count):
                write_random_png(root / subdirectory / class_name / f"{index}.png", 20, index)
    write_random_png(root / "TEST" / "chest2.png", 20, 1)
    write_random_png(root / "TEST" / "abd2.png", 20, 2)
    return root


@pytest.fixture
def dataset_archive(image_directory: Path, tmp_path: Path) -> Path:
    """A zip archive of ``image_directory``, as served by the original repository."""
    archive = tmp_path / "archive.zip"
    with zipfile.ZipFile(archive, "w") as zipped:
        for file in image_directory.rglob("*"):
            zipped.write(file, file.relative_to(image_directory.parent))
    return archive
