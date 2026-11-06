from hello_world_medical_imaging.config import Augmentation
from hello_world_medical_imaging.model import augmentation_layers, build_model
from tests.conftest import IMAGE_SIZE


def test_model_outputs_one_probability_per_image() -> None:
    model = build_model(IMAGE_SIZE, learning_rate=1e-3, pretrained=False)
    assert model.output_shape == (None, 1)
    assert model.input_shape == (None, IMAGE_SIZE, IMAGE_SIZE, 3)


def test_augmentation_keeps_image_shape() -> None:
    layers = augmentation_layers(Augmentation())
    assert layers.compute_output_shape((None, IMAGE_SIZE, IMAGE_SIZE, 3)) == (
        None,
        IMAGE_SIZE,
        IMAGE_SIZE,
        3,
    )
