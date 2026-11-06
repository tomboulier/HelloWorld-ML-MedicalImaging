"""Integration test: the whole pipeline on synthetic data, one epoch, random weights."""

from pathlib import Path

from hello_world_medical_imaging.data import synthetic_datasets
from hello_world_medical_imaging.model import build_model
from hello_world_medical_imaging.predict import predict_image
from hello_world_medical_imaging.training import plot_history, train
from tests.conftest import IMAGE_SIZE


def test_one_epoch_on_synthetic_data_runs_end_to_end(image_directory: Path, tmp_path: Path) -> None:
    train_dataset, validation_dataset = synthetic_datasets(
        IMAGE_SIZE, batch_size=4, seed=0, train_samples=8, validation_samples=4
    )
    model = build_model(IMAGE_SIZE, learning_rate=1e-3, pretrained=False)

    history = train(model, train_dataset, validation_dataset, epochs=1)

    assert len(history.epoch) == 1
    assert set(history.history) == {"loss", "accuracy", "val_loss", "val_accuracy"}

    score = predict_image(model, image_directory / "TEST" / "chest2.png", IMAGE_SIZE)
    assert 0.0 <= score <= 1.0

    assert plot_history(history, tmp_path / "curves.png").stat().st_size > 0
