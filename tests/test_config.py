from pathlib import Path

import pytest

from hello_world_medical_imaging.config import Augmentation, Config, load_config


def test_defaults_match_the_original_notebook() -> None:
    config = Config()
    assert config.image_size == 299
    assert config.batch_size == 5
    assert config.epochs == 20
    assert config.learning_rate == pytest.approx(1e-4)
    assert config.augmentation == Augmentation(20.0, 0.2, 0.2, 0.2, True)


def test_load_config_reads_toml_and_converts_paths(tmp_path: Path) -> None:
    file = tmp_path / "config.toml"
    file.write_text(
        'dataset_directory = "images"\nepochs = 2\nseed = 7\n'
        "[augmentation]\nhorizontal_flip = false\n"
    )
    config = load_config(file)
    assert config.dataset_directory == Path("images")
    assert config.epochs == 2
    assert config.seed == 7
    assert config.augmentation.horizontal_flip is False
    assert config.augmentation.zoom_range == pytest.approx(0.2)  # default kept


def test_unknown_key_is_an_error() -> None:
    with pytest.raises(TypeError):
        Config.from_dict({"learning_rat": 0.1})


def test_repository_config_file_is_valid() -> None:
    config = load_config(Path(__file__).parent.parent / "config.toml")
    assert config.epochs == 20
