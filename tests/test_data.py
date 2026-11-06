from pathlib import Path

import pytest

from hello_world_medical_imaging.data import (
    download_dataset,
    load_image_datasets,
    synthetic_datasets,
)


def test_download_dataset_unpacks_the_archive(dataset_archive: Path, tmp_path: Path) -> None:
    destination = tmp_path / "unpacked" / "Open_I_abd_vs_CXRs"

    result = download_dataset(dataset_archive.as_uri(), destination)

    assert result == destination
    assert (destination / "TRAIN" / "openI_CXR" / "0.png").exists()
    assert (destination / "TEST" / "abd2.png").exists()


def test_download_dataset_is_skipped_when_destination_exists(tmp_path: Path) -> None:
    destination = tmp_path / "existing"
    destination.mkdir()

    download_dataset("file:///does/not/exist.zip", destination)  # would fail if attempted


def test_download_dataset_rejects_archive_with_wrong_layout(tmp_path: Path) -> None:
    import zipfile

    archive = tmp_path / "wrong.zip"
    with zipfile.ZipFile(archive, "w") as zipped:
        zipped.writestr("something_else/readme.txt", "no images here")

    with pytest.raises(FileNotFoundError):
        download_dataset(archive.as_uri(), tmp_path / "expected_name")


def test_load_image_datasets_yields_batches_of_images_and_binary_labels(
    image_directory: Path,
) -> None:
    train, validation = load_image_datasets(image_directory, image_size=32, batch_size=4, seed=0)

    images, labels = next(iter(train))
    assert tuple(images.shape) == (4, 32, 32, 3)
    assert tuple(labels.shape) == (4, 1)
    assert set(labels.numpy().flatten()) <= {0.0, 1.0}
    assert sum(len(batch[1]) for batch in train) == 8
    assert sum(len(batch[1]) for batch in validation) == 4


def test_synthetic_datasets_have_the_same_contract_as_real_ones(image_directory: Path) -> None:
    real, _ = load_image_datasets(image_directory, image_size=32, batch_size=4, seed=0)
    fake, _ = synthetic_datasets(image_size=32, batch_size=4, seed=0, train_samples=8)

    assert real.element_spec[0].shape.as_list() == fake.element_spec[0].shape.as_list()
    assert real.element_spec[1].shape.as_list() == fake.element_spec[1].shape.as_list()
